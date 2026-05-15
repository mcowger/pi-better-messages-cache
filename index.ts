/**
 * pi-better-messages-cache
 *
 * Implements the dual cache-breakpoint strategy proposed in
 * https://github.com/badlogic/pi-mono/pull/1737 as a standalone pi package.
 *
 * Problem
 * -------
 * The built-in Anthropic provider places a single cache_control marker on the
 * last *user* message block.  On some providers — notably MiniMax and Kimi —
 * the preceding assistant tool_use and thinking blocks sit *outside* the cached
 * window, causing very poor cache hit rates even after many turns.
 *
 * Fix: dual-marking
 * -----------------
 * Mark TWO locations per turn:
 *   1. The last assistant tool_use block  ← this extension adds this
 *   2. The last user message block        ← the built-in provider already does this
 *
 * Both markers together ensure the full assistant-turn (thinking + tool_use +
 * tool_result) sits inside the growing cached prefix.
 *
 * This aligns with the cache strategy used by OpenCode, Kilo Code, and Roo Code.
 *
 * How it works
 * ------------
 * The extension calls pi.registerProvider("anthropic", { api: "anthropic-messages",
 * streamSimple }) at load time.  Because the api-registry is keyed by API type
 * string, this transparently replaces the built-in handler for every model that
 * uses the anthropic-messages API — all native Anthropic models — while leaving
 * model definitions, pricing, OAuth config, and every other setting completely
 * untouched.
 *
 * Empirical impact (field data from PR #1737):
 *   - MiniMax / Kimi providers : cache hit rates often reach 80 %+ instead of
 *     stalling near the system-prompt-sized read window.
 *   - Anthropic native models  : small positive improvement.
 *
 * Installation
 * ------------
 *   pi install npm:@mcowger/pi-better-messages-cache
 */

import Anthropic from "@anthropic-ai/sdk";
import type {
	ContentBlockParam,
	MessageCreateParamsStreaming,
} from "@anthropic-ai/sdk/resources/messages.js";
import {
	calculateCost,
	createAssistantMessageEventStream,
	parseJsonWithRepair,
	parseStreamingJson,
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type StopReason,
	type TextContent,
	type ThinkingContent,
	type Tool,
	type ToolCall,
	type ToolResultMessage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// SSE parsing — mirrors the built-in pi-ai Anthropic provider so that
// parseJsonWithRepair is used instead of the SDK's bare JSON.parse.
//
// The Anthropic SDK's stream() method parses each SSE event with a plain
// JSON.parse. When the model emits raw \t or \n inside a tool-call JSON string
// (e.g. tab-indented oldText in an Edit call), the wire event looks like:
//   data: {"delta":{"type":"input_json_delta","partial_json":"\t\tenv: {"}}
// JSON.parse throws "Bad control character in string literal in JSON at
// position N" which propagates out of the for-await loop, cuts the stream
// before content_block_stop fires, and leaves arguments as {}.
// The error is displayed as the tool result and cannot be retried because
// the model has no context about what the original arguments were.
//
// Fix: use client.messages.create().asResponse() to get the raw HTTP
// response and parse SSE events ourselves with parseJsonWithRepair, which
// escapes raw control chars before handing off to JSON.parse.
// ---------------------------------------------------------------------------

interface SseEvent {
	event: string | null;
	data: string;
	raw: string[];
}

interface SseDecoderState {
	event: string | null;
	data: string[];
	raw: string[];
}

const ANTHROPIC_STREAM_EVENTS = new Set([
	"message_start",
	"message_delta",
	"message_stop",
	"content_block_start",
	"content_block_delta",
	"content_block_stop",
	"ping",
]);

async function* iterateSseMessages(
	body: ReadableStream<Uint8Array>,
	signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
	const reader = body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	let state: SseDecoderState = { event: null, data: [], raw: [] };

	function flush(): SseEvent | null {
		if (state.data.length === 0) return null;
		return { event: state.event, data: state.data.join("\n"), raw: state.raw };
	}

	try {
		while (true) {
			if (signal?.aborted) break;
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";
			for (const line of lines) {
				state.raw.push(line);
				if (line === "") {
					const ev = flush();
					if (ev) yield ev;
					state = { event: null, data: [], raw: [] };
				} else if (line.startsWith("event:")) {
					state.event = line.slice(6).trim();
				} else if (line.startsWith("data:")) {
					state.data.push(line.slice(5).trim());
				}
			}
		}
		const trailing = flush();
		if (trailing) yield trailing;
	} finally {
		reader.releaseLock();
	}
}

async function* iterateAnthropicSseEvents(
	response: Response,
	signal?: AbortSignal,
): AsyncGenerator<any> {
	if (!response.body) throw new Error("Anthropic response has no body");
	for await (const sse of iterateSseMessages(response.body, signal)) {
		if (sse.event === "error") throw new Error(sse.data);
		if (!ANTHROPIC_STREAM_EVENTS.has(sse.event ?? "")) continue;
		try {
			// parseJsonWithRepair escapes raw control chars (\t, \n, etc.) inside
			// JSON string literals before parsing — the SDK's JSON.parse cannot.
			yield parseJsonWithRepair(sse.data);
		} catch (error) {
			const msg = error instanceof Error ? error.message : String(error);
			throw new Error(`Could not parse Anthropic SSE event "${sse.event}": ${msg}`);
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Claude Code canonical tool names used in OAuth / stealth mode. */
const CLAUDE_CODE_TOOLS = [
	"Read",
	"Write",
	"Edit",
	"Bash",
	"Grep",
	"Glob",
	"AskUserQuestion",
	"TodoWrite",
	"WebFetch",
	"WebSearch",
];
const ccToolLookup = new Map(CLAUDE_CODE_TOOLS.map((t) => [t.toLowerCase(), t]));

function toClaudeCodeName(name: string): string {
	return ccToolLookup.get(name.toLowerCase()) ?? name;
}

function fromClaudeCodeName(name: string, tools?: Tool[]): string {
	const lower = name.toLowerCase();
	return tools?.find((t) => t.name.toLowerCase() === lower)?.name ?? name;
}

function isOAuthToken(apiKey: string): boolean {
	return apiKey.includes("sk-ant-oat");
}

/** Replace lone surrogate code points so the Anthropic API never sees them. */
function sanitizeSurrogates(text: string): string {
	return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

/**
 * Convert tool-result content blocks into the form expected by the Anthropic API.
 * Plain-text-only content is flattened to a single string; mixed content retains
 * the full block array.
 */
function convertContentBlocks(
	content: (TextContent | ImageContent)[],
): string | Array<{ type: "text"; text: string } | { type: "image"; source: unknown }> {
	const hasImages = content.some((c) => c.type === "image");
	if (!hasImages) {
		return sanitizeSurrogates(content.map((c) => (c as TextContent).text).join("\n"));
	}
	const blocks = content.map((block) => {
		if (block.type === "text") {
			return { type: "text" as const, text: sanitizeSurrogates(block.text) };
		}
		const img = block as ImageContent;
		return {
			type: "image" as const,
			source: { type: "base64" as const, media_type: img.mimeType, data: img.data },
		};
	});
	// Anthropic requires at least one text block before any image block.
	if (!blocks.some((b) => b.type === "text")) {
		blocks.unshift({ type: "text" as const, text: "(see attached image)" });
	}
	return blocks;
}

// ---------------------------------------------------------------------------
// convertMessages — dual cache-breakpoint strategy
// ---------------------------------------------------------------------------

type CacheControl = { type: "ephemeral" };

/**
 * Convert the internal Message[] representation to the Anthropic API's
 * MessageParam[] format, applying the dual cache-breakpoint strategy:
 *
 *   1. The **last assistant tool_use block** is marked with cache_control.
 *      This pulls the entire assistant turn (thinking blocks + all tool_use
 *      blocks) inside the growing cached prefix on providers that extend the
 *      window backward from the marked block.
 *
 *   2. The **last user message block** is also marked with cache_control
 *      (the original single-breakpoint behaviour that already existed in the
 *      built-in provider, preserved here).
 *
 * Without marker (1), providers such as MiniMax and Kimi only cache up to the
 * tool_result in the user message, leaving the assistant blocks outside the
 * window and causing cache misses on every subsequent turn.
 */
export function convertMessages(
	messages: Message[],
	isOAuth: boolean,
	cacheControl: CacheControl,
	tools?: Tool[],
): any[] {
	const params: any[] = [];

	for (let i = 0; i < messages.length; i++) {
		const msg = messages[i];

		// If this is an assistant message with tool_use blocks, check that the
		// next message contains matching tool_results. If not (aborted turn,
		// steering injection, etc.), insert synthetic error tool_results so the
		// API never sees an unmatched tool_use.
		if (msg.role === "assistant") {
			const toolCalls = (msg.content as any[]).filter((b) => b.type === "toolCall");
			if (toolCalls.length > 0) {
				const next = messages[i + 1];
				const nextIsToolResults =
					next?.role === "toolResult" ||
					(next?.role === "user" && Array.isArray(next.content) &&
						(next.content as any[]).every((b) => b.type === "tool_result"));
				if (!nextIsToolResults) {
					// Inject synthetic tool_result messages for each orphaned tool_use
					const synthetics: Message[] = toolCalls.map((tc) => ({
						role: "toolResult" as const,
						toolCallId: tc.id,
						toolName: tc.name,
						content: [{ type: "text", text: "No result: tool call was interrupted" }],
						isError: true,
						timestamp: Date.now(),
					}));
					messages = [
						...messages.slice(0, i + 1),
						...synthetics,
						...messages.slice(i + 1),
					];
				}
			}
		}

		// ── user message ────────────────────────────────────────────────────
		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				if (msg.content.trim().length > 0) {
					params.push({ role: "user", content: sanitizeSurrogates(msg.content) });
				}
			} else {
				const blocks: ContentBlockParam[] = (msg.content as any[]).flatMap((item) => {
					// Pass tool_result blocks through unchanged (already in Anthropic format)
					if ((item as any).type === "tool_result") return [item];
					if (item.type === "text") {
						const text = sanitizeSurrogates(item.text);
						return text.trim().length > 0 ? [{ type: "text" as const, text }] : [];
					}
					const img = item as ImageContent;
					return [{ type: "image" as const, source: { type: "base64" as const, media_type: img.mimeType as any, data: img.data } }];
				});
				if (blocks.length > 0) {
					params.push({ role: "user", content: blocks });
				}
			}

		// ── assistant message ────────────────────────────────────────────────
		} else if (msg.role === "assistant") {
			const blocks: ContentBlockParam[] = [];

			// ----------------------------------------------------------------
			// Dual-cache change (1 of 2):
			// Identify the last toolCall block so we can stamp it with
			// cache_control.  This anchors the cache window to include all
			// earlier blocks in the same assistant turn (thinking, text, and
			// preceding tool_use blocks).
			// ----------------------------------------------------------------
			const content = msg.content as any[];
			const lastToolCallIndex = content.map((b) => b.type).lastIndexOf("toolCall");

			for (const [idx, block] of content.entries()) {
				if (block.type === "text") {
					if (block.text.trim().length === 0) continue;
					blocks.push({ type: "text", text: sanitizeSurrogates(block.text) });

				} else if (block.type === "thinking") {
					if (block.redacted) {
						// Redacted thinking — pass opaque payload back as-is
						blocks.push({ type: "redacted_thinking" as any, data: block.thinkingSignature });
						continue;
					}
					if (block.thinking.trim().length === 0) continue;
					if (!block.thinkingSignature || block.thinkingSignature.trim().length === 0) {
						// Aborted / missing signature: demote to plain text to
						// avoid API rejection and prevent the model imitating tags
						blocks.push({ type: "text", text: sanitizeSurrogates(block.thinking) });
					} else {
						blocks.push({
							type: "thinking" as any,
							thinking: sanitizeSurrogates(block.thinking),
							signature: block.thinkingSignature,
						});
					}

				} else if (block.type === "toolCall") {
					blocks.push({
						type: "tool_use",
						id: block.id,
						name: isOAuth ? toClaudeCodeName(block.name) : block.name,
						input: block.arguments ?? {},
						// ----------------------------------------------------------
						// Dual-cache change (1 of 2):
						// Mark the last tool_use block with cache_control so
						// that the assistant turn is included in the cached
						// prefix on every subsequent call.
						// ----------------------------------------------------------
						...(idx === lastToolCallIndex ? { cache_control: cacheControl } : {}),
					});
				}
			}

			if (blocks.length > 0) {
				params.push({ role: "assistant", content: blocks });
			}

		// ── tool results ─────────────────────────────────────────────────────
		} else if (msg.role === "toolResult") {
			// Collect all consecutive toolResult messages into a single user turn,
			// matching the behaviour required by the z.ai Anthropic endpoint.
			const toolResults: any[] = [];

			toolResults.push({
				type: "tool_result",
				tool_use_id: (msg as ToolResultMessage).toolCallId,
				content: convertContentBlocks((msg as ToolResultMessage).content),
				is_error: (msg as ToolResultMessage).isError,
			});

			let j = i + 1;
			while (j < messages.length && messages[j].role === "toolResult") {
				const next = messages[j] as ToolResultMessage;
				toolResults.push({
					type: "tool_result",
					tool_use_id: next.toolCallId,
					content: convertContentBlocks(next.content),
					is_error: next.isError,
				});
				j++;
			}

			i = j - 1; // skip lookahead messages in outer loop
			params.push({ role: "user", content: toolResults });
		}
	}

	// -------------------------------------------------------------------------
	// Dual-cache change (2 of 2):
	// Mark the last block of the last user message with cache_control.
	// This is the original single-breakpoint pattern from the built-in provider;
	// it is preserved here and works in tandem with the tool_use marker above.
	// -------------------------------------------------------------------------
	if (params.length > 0) {
		const last = params[params.length - 1];
		if (last.role === "user") {
			if (Array.isArray(last.content)) {
				const lastBlock = last.content[last.content.length - 1];
				if (
					lastBlock &&
					(lastBlock.type === "text" ||
						lastBlock.type === "image" ||
						lastBlock.type === "tool_result")
				) {
					lastBlock.cache_control = cacheControl;
				}
			} else if (typeof last.content === "string") {
				// Promote string content to a block array so we can attach cache_control
				last.content = [
					{ type: "text", text: last.content, cache_control: cacheControl },
				];
			}
		}
	}

	// -------------------------------------------------------------------------
	// Merge consecutive user-role params into one.
	// This handles all cases where pi emits back-to-back user messages:
	//   - tool results followed by a steering message
	//   - synthetic tool results (from transformMessages) before a steering message
	//   - aborted assistant stripped by transformMessages leaving tool results
	//     adjacent to the next user message
	//   - multiple steering messages queued during tool execution
	// -------------------------------------------------------------------------
	const merged: any[] = [];
	for (const param of params) {
		const prev = merged[merged.length - 1];
		if (prev && prev.role === "user" && param.role === "user") {
			// Flatten both sides to block arrays and concatenate
			const prevBlocks: any[] = Array.isArray(prev.content)
				? prev.content
				: [{ type: "text", text: prev.content }];
			const nextBlocks: any[] = Array.isArray(param.content)
				? param.content
				: [{ type: "text", text: param.content }];
			prev.content = [...prevBlocks, ...nextBlocks];
		} else {
			merged.push(param);
		}
	}

	return merged;
}

function convertTools(tools: Tool[], isOAuth: boolean): any[] {
	return tools.map((tool) => ({
		name: isOAuth ? toClaudeCodeName(tool.name) : tool.name,
		description: tool.description,
		input_schema: {
			type: "object",
			properties: (tool.parameters as any).properties ?? {},
			required: (tool.parameters as any).required ?? [],
		},
	}));
}

function mapStopReason(reason: string): StopReason {
	switch (reason) {
		case "end_turn":
		case "pause_turn":
		case "stop_sequence":
			return "stop";
		case "max_tokens":
			return "length";
		case "tool_use":
			return "toolUse";
		default:
			return "error";
	}
}

/**
 * Anthropic allows at most 4 blocks with cache_control across the entire
 * request payload (system + messages). Preserve the newest message-level
 * breakpoints and trim older ones first; keep system markers intact.
 */
function enforceCacheControlLimit(
	params: Pick<MessageCreateParamsStreaming, "messages" | "system">,
	maxBreakpoints = 4,
): void {
	const systemBlocks = Array.isArray(params.system) ? params.system : [];
	const systemMarkerCount = systemBlocks.reduce(
		(count, block: any) => count + (block?.cache_control ? 1 : 0),
		0,
	);

	const messageMarkers: any[] = [];
	for (const message of params.messages ?? []) {
		if (!Array.isArray((message as any).content)) continue;
		for (const block of (message as any).content) {
			if (block?.cache_control) {
				messageMarkers.push(block);
			}
		}
	}

	const totalMarkers = systemMarkerCount + messageMarkers.length;
	if (totalMarkers <= maxBreakpoints) return;

	const markersToRemove = totalMarkers - maxBreakpoints;
	for (const block of messageMarkers.slice(0, markersToRemove)) {
		delete block.cache_control;
	}
}

// ---------------------------------------------------------------------------
// Streaming implementation
// ---------------------------------------------------------------------------

/**
 * Drop-in replacement for the built-in anthropic-messages stream handler.
 *
 * Identical to the built-in provider except that convertMessages() applies
 * the dual cache-breakpoint strategy described above.
 */
function streamWithDualCacheBreakpoints(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const apiKey = options?.apiKey ?? "";
			const isOAuth = isOAuthToken(apiKey);
			const cacheControl: CacheControl = { type: "ephemeral" };

			const betaFeatures = [
				"fine-grained-tool-streaming-2025-05-14",
				"interleaved-thinking-2025-05-14",
			];

			// Build Anthropic client
			const clientOptions: any = {
				baseURL: model.baseUrl,
				dangerouslyAllowBrowser: true,
			};

			if (isOAuth) {
				clientOptions.apiKey = null;
				clientOptions.authToken = apiKey;
				clientOptions.defaultHeaders = {
					accept: "application/json",
					"anthropic-dangerous-direct-browser-access": "true",
					"anthropic-beta": `claude-code-20250219,oauth-2025-04-20,${betaFeatures.join(",")}`,
					"user-agent": "claude-cli/2.1.2 (external, cli)",
					"x-app": "cli",
					...(model.headers ?? {}),
				};
			} else {
				clientOptions.apiKey = apiKey;
				clientOptions.defaultHeaders = {
					accept: "application/json",
					"anthropic-dangerous-direct-browser-access": "true",
					"anthropic-beta": betaFeatures.join(","),
					...(model.headers ?? {}),
				};
			}

			const client = new Anthropic(clientOptions);

			// Build request parameters
			const params: MessageCreateParamsStreaming = {
				model: model.id,
				messages: convertMessages(context.messages, isOAuth, cacheControl, context.tools),
				max_tokens: options?.maxTokens ?? Math.floor(model.maxTokens / 3),
				stream: true,
			};

			// System prompt (with cache_control on each block)
			if (isOAuth) {
				params.system = [
					{
						type: "text",
						text: "You are Claude Code, Anthropic's official CLI for Claude.",
						cache_control: cacheControl,
					},
				];
				if (context.systemPrompt) {
					params.system.push({
						type: "text",
						text: sanitizeSurrogates(context.systemPrompt),
						cache_control: cacheControl,
					});
				}
			} else if (context.systemPrompt) {
				params.system = [
					{
						type: "text",
						text: sanitizeSurrogates(context.systemPrompt),
						cache_control: cacheControl,
					},
				];
			}

			// Tools
			if (context.tools && context.tools.length > 0) {
				params.tools = convertTools(context.tools, isOAuth);
			}

			enforceCacheControlLimit(params);

			// Extended thinking / reasoning
			if (options?.reasoning && model.reasoning) {
				const defaultBudgets: Record<string, number> = {
					minimal: 1024,
					low: 4096,
					medium: 10240,
					high: 20480,
					xhigh: 32000,
				};
				const budget =
					options.thinkingBudgets?.[options.reasoning as keyof typeof options.thinkingBudgets] ??
					defaultBudgets[options.reasoning] ??
					10240;
				(params as any).thinking = { type: "enabled", budget_tokens: budget };
			}

			// Fire onPayload before the network call so tests can capture the full
			// request payload without needing a real API key.
			options?.onPayload?.(params);

			// Stream — use raw HTTP + custom SSE parser instead of SDK stream().
			// See the SSE parsing section above for the full explanation.
			const httpResponse = await (client.messages.create as any)(
				params,
				{ signal: options?.signal },
			).asResponse();

			stream.push({ type: "start", partial: output });

			// We use a sentinel `index` property on each block to correlate stream
			// events with output.content entries before content_block_stop fires.
			type BlockWithIndex = (ThinkingContent | TextContent | (ToolCall & { partialJson: string })) & {
				index: number;
			};
			const blocks = output.content as BlockWithIndex[];

			for await (const event of iterateAnthropicSseEvents(httpResponse, options?.signal)) {
				if (event.type === "message_start") {
					output.usage.input = event.message.usage.input_tokens ?? 0;
					output.usage.output = event.message.usage.output_tokens ?? 0;
					output.usage.cacheRead = (event.message.usage as any).cache_read_input_tokens ?? 0;
					output.usage.cacheWrite =
						(event.message.usage as any).cache_creation_input_tokens ?? 0;
					output.usage.totalTokens =
						output.usage.input +
						output.usage.output +
						output.usage.cacheRead +
						output.usage.cacheWrite;
					calculateCost(model, output.usage);
				} else if (event.type === "content_block_start") {
					if (event.content_block.type === "text") {
						output.content.push({ type: "text", text: "", index: event.index } as any);
						stream.push({
							type: "text_start",
							contentIndex: output.content.length - 1,
							partial: output,
						});
					} else if (event.content_block.type === "thinking") {
						output.content.push({
							type: "thinking",
							thinking: "",
							thinkingSignature: "",
							index: event.index,
						} as any);
						stream.push({
							type: "thinking_start",
							contentIndex: output.content.length - 1,
							partial: output,
						});
					} else if (event.content_block.type === "tool_use") {
						output.content.push({
							type: "toolCall",
							id: event.content_block.id,
							name: isOAuth
								? fromClaudeCodeName(event.content_block.name, context.tools)
								: event.content_block.name,
							arguments: {},
							partialJson: "",
							index: event.index,
						} as any);
						stream.push({
							type: "toolcall_start",
							contentIndex: output.content.length - 1,
							partial: output,
						});
					}
				} else if (event.type === "content_block_delta") {
					const pos = blocks.findIndex((b) => b.index === event.index);
					const block = blocks[pos];
					if (!block) continue;

					if (event.delta.type === "text_delta" && block.type === "text") {
						block.text += event.delta.text;
						stream.push({
							type: "text_delta",
							contentIndex: pos,
							delta: event.delta.text,
							partial: output,
						});
					} else if (event.delta.type === "thinking_delta" && block.type === "thinking") {
						block.thinking += event.delta.thinking;
						stream.push({
							type: "thinking_delta",
							contentIndex: pos,
							delta: event.delta.thinking,
							partial: output,
						});
					} else if (
						event.delta.type === "input_json_delta" &&
						block.type === "toolCall"
					) {
						(block as any).partialJson += event.delta.partial_json;
						block.arguments = parseStreamingJson((block as any).partialJson);
						stream.push({
							type: "toolcall_delta",
							contentIndex: pos,
							delta: event.delta.partial_json,
							partial: output,
						});
					} else if (
						event.delta.type === "signature_delta" &&
						block.type === "thinking"
					) {
						block.thinkingSignature =
							(block.thinkingSignature ?? "") + (event.delta as any).signature;
					}
				} else if (event.type === "content_block_stop") {
					const pos = blocks.findIndex((b) => b.index === event.index);
					const block = blocks[pos];
					if (!block) continue;

					delete (block as any).index;

					if (block.type === "text") {
						stream.push({
							type: "text_end",
							contentIndex: pos,
							content: block.text,
							partial: output,
						});
					} else if (block.type === "thinking") {
						stream.push({
							type: "thinking_end",
							contentIndex: pos,
							content: block.thinking,
							partial: output,
						});
					} else if (block.type === "toolCall") {
						block.arguments = parseStreamingJson((block as any).partialJson);
						delete (block as any).partialJson;
						stream.push({
							type: "toolcall_end",
							contentIndex: pos,
							toolCall: block,
							partial: output,
						});
					}
				} else if (event.type === "message_delta") {
					if ((event.delta as any).stop_reason) {
						output.stopReason = mapStopReason((event.delta as any).stop_reason);
					}
					output.usage.input = (event.usage as any).input_tokens ?? output.usage.input;
					output.usage.output = (event.usage as any).output_tokens ?? output.usage.output;
					output.usage.cacheRead =
						(event.usage as any).cache_read_input_tokens ?? output.usage.cacheRead;
					output.usage.cacheWrite =
						(event.usage as any).cache_creation_input_tokens ?? output.usage.cacheWrite;
					output.usage.totalTokens =
						output.usage.input +
						output.usage.output +
						output.usage.cacheRead +
						output.usage.cacheWrite;
					calculateCost(model, output.usage);
				}
			}

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}

			stream.push({
				type: "done",
				reason: output.stopReason as "stop" | "length" | "toolUse",
				message: output,
			});
			stream.end();
		} catch (error) {
			// Clean up sentinel index fields before emitting the error message
			for (const block of output.content) delete (block as any).index;
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage =
				error instanceof Error ? error.message : JSON.stringify(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI): void {
	/**
	 * Register a custom streamSimple handler for the "anthropic-messages" API.
	 *
	 * Providing only `api` + `streamSimple` (no `models`) replaces the api-registry
	 * entry for "anthropic-messages" globally, affecting every model that uses that
	 * API type, while leaving all model definitions, pricing tables, and OAuth
	 * configuration completely untouched.
	 */
	pi.registerProvider("anthropic", {
		api: "anthropic-messages",
		streamSimple: streamWithDualCacheBreakpoints,
	});
}
