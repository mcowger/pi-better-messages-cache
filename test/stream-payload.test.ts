/**
 * Integration tests for streamWithDualCacheBreakpoints() via onPayload.
 *
 * These tests fire the stream function with a fake API key.  The request
 * fails at the network layer, but `options.onPayload` is called before the
 * Anthropic SDK makes the actual HTTP call, so we can capture and assert on
 * the full MessageCreateParams payload without any network access.
 *
 * Live API tests (skipped unless ANTHROPIC_API_KEY is set) verify end-to-end
 * streaming works correctly and that cache tokens are actually reported back.
 */

import { describe, expect, it } from "vitest";

// We need a Model-shaped object.  Build a minimal one that satisfies the
// fields our stream function reads.
function fakeModel(overrides: Record<string, any> = {}): any {
	return {
		id: "claude-sonnet-4-20250514",
		name: "Claude Sonnet",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "https://api.anthropic.com",
		reasoning: false,
		input: ["text"],
		maxTokens: 16000,
		contextWindow: 200000,
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		...overrides,
	};
}

// Dynamically import the stream function via the default-export factory so
// we don't need to publish it as a separate named export.
// We reach into the module's named export instead.
async function getStreamFn() {
	// The default export registers the provider; the stream function is
	// invoked internally.  For testing we import the module directly and
	// call streamWithDualCacheBreakpoints via the module boundary.
	// Because it's not a named export we test payload behaviour via onPayload.
	const mod = await import("../index.js");
	return mod;
}

// ---------------------------------------------------------------------------
// Helper: run the stream until the first event (or error) and capture payload
// ---------------------------------------------------------------------------

async function capturePayload(
	context: any,
	modelOverrides: Record<string, any> = {},
	options: Record<string, any> = {},
): Promise<any> {
	// We need to call streamWithDualCacheBreakpoints directly.
	// It is not exported by name, but we can indirectly invoke it by
	// constructing a minimal ExtensionAPI mock, running the factory, and
	// retrieving the registered streamSimple.
	let registeredStreamFn: Function | null = null;

	const mockPi: any = {
		registerProvider(_name: string, config: any) {
			registeredStreamFn = config.streamSimple;
		},
	};

	const { default: factory } = await import("../index.js");
	factory(mockPi);

	if (!registeredStreamFn) throw new Error("streamSimple was not registered");

	let capturedPayload: any = null;
	const model = fakeModel(modelOverrides);

	const stream = (registeredStreamFn as Function)(model, context, {
		apiKey: "fake-sk-ant-key",
		...options,
		onPayload: (p: unknown) => {
			capturedPayload = p;
		},
	});

	// Consume until we have the payload or hit an error event
	for await (const event of stream) {
		if (capturedPayload !== null) break;
		if (event.type === "error") break;
	}

	return capturedPayload;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("stream payload — system prompt", () => {
	it("adds cache_control to the system prompt block", async () => {
		const context = {
			systemPrompt: "You are helpful.",
			messages: [{ role: "user", content: "hi", timestamp: 0 }],
		};
		const payload = await capturePayload(context);

		expect(payload).not.toBeNull();
		expect(payload.system).toBeDefined();
		expect(payload.system[0].text).toBe("You are helpful.");
		expect(payload.system[0].cache_control).toEqual({ type: "ephemeral" });
	});

	it("omits system when no systemPrompt is provided", async () => {
		const context = {
			messages: [{ role: "user", content: "hi", timestamp: 0 }],
		};
		const payload = await capturePayload(context);

		expect(payload.system).toBeUndefined();
	});
});

describe("stream payload — user message cache_control (dual-cache part 2)", () => {
	it("places cache_control on the last block of the last user message", async () => {
		const context = {
			systemPrompt: "sys",
			messages: [{ role: "user", content: "hello world", timestamp: 0 }],
		};
		const payload = await capturePayload(context);

		const msgs: any[] = payload.messages;
		const lastMsg = msgs[msgs.length - 1];
		expect(lastMsg.role).toBe("user");
		const lastBlock = lastMsg.content[lastMsg.content.length - 1];
		expect(lastBlock.cache_control).toEqual({ type: "ephemeral" });
	});
});

describe("stream payload — assistant tool_use cache_control (dual-cache part 1)", () => {
	it("places cache_control on the last tool_use block in an assistant message", async () => {
		const context = {
			messages: [
				{ role: "user", content: "run something", timestamp: 0 },
				{
					role: "assistant",
					content: [
						{ type: "toolCall", id: "tc1", name: "bash", arguments: { command: "ls" } },
					],
					api: "anthropic-messages",
					provider: "anthropic",
					model: "claude-sonnet-4-20250514",
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
					stopReason: "toolUse",
					timestamp: 0,
				},
				{ role: "toolResult", toolCallId: "tc1", toolName: "bash", content: [{ type: "text", text: "file.txt" }], isError: false, timestamp: 0 },
			],
		};
		const payload = await capturePayload(context);

		const assistantMsg = payload.messages.find((m: any) => m.role === "assistant");
		expect(assistantMsg).toBeDefined();

		const toolUse = assistantMsg.content.find((b: any) => b.type === "tool_use");
		expect(toolUse).toBeDefined();
		expect(toolUse.cache_control).toEqual({ type: "ephemeral" });
	});

	it("marks only the LAST tool_use when the assistant message has multiple", async () => {
		const context = {
			messages: [
				{ role: "user", content: "do two things", timestamp: 0 },
				{
					role: "assistant",
					content: [
						{ type: "toolCall", id: "tc1", name: "read", arguments: {} },
						{ type: "toolCall", id: "tc2", name: "bash", arguments: { command: "echo hi" } },
					],
					api: "anthropic-messages",
					provider: "anthropic",
					model: "claude-sonnet-4-20250514",
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
					stopReason: "toolUse",
					timestamp: 0,
				},
				{ role: "toolResult", toolCallId: "tc1", toolName: "read", content: [{ type: "text", text: "r1" }], isError: false, timestamp: 0 },
				{ role: "toolResult", toolCallId: "tc2", toolName: "bash", content: [{ type: "text", text: "hi" }], isError: false, timestamp: 0 },
			],
		};
		const payload = await capturePayload(context);

		const assistantMsg = payload.messages.find((m: any) => m.role === "assistant");
		const toolUses = assistantMsg.content.filter((b: any) => b.type === "tool_use");
		expect(toolUses).toHaveLength(2);
		expect(toolUses[0].cache_control).toBeUndefined();
		expect(toolUses[1].cache_control).toEqual({ type: "ephemeral" });
	});
});

describe("stream payload — cache_control limit", () => {
	it("trims older message cache markers so the request never exceeds four total", async () => {
		const context = {
			systemPrompt: "sys",
			messages: [
				{ role: "user", content: "turn 1", timestamp: 0 },
				{
					role: "assistant",
					content: [{ type: "toolCall", id: "tc1", name: "bash", arguments: { command: "echo 1" } }],
					api: "anthropic-messages",
					provider: "anthropic",
					model: "claude-sonnet-4-20250514",
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
					stopReason: "toolUse",
					timestamp: 0,
				},
				{ role: "toolResult", toolCallId: "tc1", toolName: "bash", content: [{ type: "text", text: "r1" }], isError: false, timestamp: 0 },
				{
					role: "assistant",
					content: [{ type: "toolCall", id: "tc2", name: "bash", arguments: { command: "echo 2" } }],
					api: "anthropic-messages",
					provider: "anthropic",
					model: "claude-sonnet-4-20250514",
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
					stopReason: "toolUse",
					timestamp: 0,
				},
				{ role: "toolResult", toolCallId: "tc2", toolName: "bash", content: [{ type: "text", text: "r2" }], isError: false, timestamp: 0 },
				{
					role: "assistant",
					content: [{ type: "toolCall", id: "tc3", name: "bash", arguments: { command: "echo 3" } }],
					api: "anthropic-messages",
					provider: "anthropic",
					model: "claude-sonnet-4-20250514",
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
					stopReason: "toolUse",
					timestamp: 0,
				},
				{ role: "toolResult", toolCallId: "tc3", toolName: "bash", content: [{ type: "text", text: "r3" }], isError: false, timestamp: 0 },
			],
		};
		const payload = await capturePayload(context);

		const allCacheMarkedBlocks = [
			...(payload.system ?? []),
			...payload.messages.flatMap((m: any) => Array.isArray(m.content) ? m.content : []),
		].filter((b: any) => b?.cache_control);

		expect(allCacheMarkedBlocks).toHaveLength(4);

		const assistantMsgs = payload.messages.filter((m: any) => m.role === "assistant");
		expect(assistantMsgs[0].content[0].cache_control).toBeUndefined();
		expect(assistantMsgs[1].content[0].cache_control).toEqual({ type: "ephemeral" });
		expect(assistantMsgs[2].content[0].cache_control).toEqual({ type: "ephemeral" });

		const userMsgs = payload.messages.filter((m: any) => m.role === "user");
		expect(userMsgs[0].content[0].cache_control).toBeUndefined();
		const lastUserLastBlock = userMsgs[userMsgs.length - 1].content[userMsgs[userMsgs.length - 1].content.length - 1];
		expect(lastUserLastBlock.cache_control).toEqual({ type: "ephemeral" });
	});
});

describe("stream payload — general structure", () => {
	it("includes the correct model id", async () => {
		const context = {
			messages: [{ role: "user", content: "hi", timestamp: 0 }],
		};
		const payload = await capturePayload(context);
		expect(payload.model).toBe("claude-sonnet-4-20250514");
	});

	it("includes stream: true", async () => {
		const context = {
			messages: [{ role: "user", content: "hi", timestamp: 0 }],
		};
		const payload = await capturePayload(context);
		expect(payload.stream).toBe(true);
	});

	it("includes tools in the payload when provided", async () => {
		const { Type } = await import("@sinclair/typebox");
		const context = {
			messages: [{ role: "user", content: "hi", timestamp: 0 }],
			tools: [
				{
					name: "bash",
					description: "Run a shell command",
					parameters: Type.Object({
						command: Type.String({ description: "Command to run" }),
					}),
				},
			],
		};
		const payload = await capturePayload(context);
		expect(payload.tools).toBeDefined();
		expect(payload.tools).toHaveLength(1);
		expect(payload.tools[0].name).toBe("bash");
	});

	it("omits tools when the context has no tools", async () => {
		const context = {
			messages: [{ role: "user", content: "hi", timestamp: 0 }],
		};
		const payload = await capturePayload(context);
		expect(payload.tools).toBeUndefined();
	});

	it("respects a custom maxTokens option", async () => {
		const context = {
			messages: [{ role: "user", content: "hi", timestamp: 0 }],
		};
		const payload = await capturePayload(context, {}, { maxTokens: 512 });
		expect(payload.max_tokens).toBe(512);
	});
});

describe("stream payload — OAuth / Claude Code mode", () => {
	it("adds Claude Code identity block when using an OAuth token", async () => {
		const context = {
			systemPrompt: "Be helpful.",
			messages: [{ role: "user", content: "hi", timestamp: 0 }],
		};
		// Use a fake OAuth-style token (contains sk-ant-oat)
		const payload = await capturePayload(context, {}, { apiKey: "sk-ant-oat-fake-token" });

		expect(payload.system).toBeDefined();
		expect(payload.system[0].text).toBe("You are Claude Code, Anthropic's official CLI for Claude.");
		expect(payload.system[0].cache_control).toEqual({ type: "ephemeral" });
		expect(payload.system[1].text).toBe("Be helpful.");
	});

	it("normalises tool names to Claude Code PascalCase in OAuth mode", async () => {
		const context = {
			messages: [
				{ role: "user", content: "read a file", timestamp: 0 },
				{
					role: "assistant",
					content: [{ type: "toolCall", id: "tc1", name: "read", arguments: { path: "/tmp/f" } }],
					api: "anthropic-messages", provider: "anthropic",
					model: "claude-sonnet-4-20250514",
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
					stopReason: "toolUse", timestamp: 0,
				},
				{ role: "toolResult", toolCallId: "tc1", toolName: "read", content: [{ type: "text", text: "data" }], isError: false, timestamp: 0 },
			],
		};
		const payload = await capturePayload(context, {}, { apiKey: "sk-ant-oat-fake-token" });

		const assistantMsg = payload.messages.find((m: any) => m.role === "assistant");
		const toolUse = assistantMsg.content.find((b: any) => b.type === "tool_use");
		expect(toolUse.name).toBe("Read");
	});
});

// ---------------------------------------------------------------------------
// Live API tests (skipped unless ANTHROPIC_API_KEY is set)
// ---------------------------------------------------------------------------

describe.skipIf(!process.env.ANTHROPIC_API_KEY)("live API — end-to-end streaming", () => {
	it("completes a simple turn and reports usage", async () => {
		let registeredStreamFn: Function | null = null;
		const mockPi: any = {
			registerProvider(_: string, config: any) {
				registeredStreamFn = config.streamSimple;
			},
		};
		const { default: factory } = await import("../index.js");
		factory(mockPi);
		if (!registeredStreamFn) throw new Error("no streamSimple");

		const model = fakeModel();
		const context = {
			systemPrompt: "Reply in one word.",
			messages: [{ role: "user", content: "Say hello.", timestamp: Date.now() }],
		};

		const stream = (registeredStreamFn as Function)(model, context, {
			apiKey: process.env.ANTHROPIC_API_KEY,
		});

		let done = false;
		let lastMessage: any = null;
		for await (const event of stream) {
			if (event.type === "done") {
				done = true;
				lastMessage = event.message;
			}
		}

		expect(done).toBe(true);
		expect(lastMessage.usage.input).toBeGreaterThan(0);
		expect(lastMessage.usage.output).toBeGreaterThan(0);
		expect(lastMessage.stopReason).toBe("stop");
	});

	it("reports cache_write tokens on the first call and cache_read on repeat", async () => {
		let registeredStreamFn: Function | null = null;
		const mockPi: any = {
			registerProvider(_: string, config: any) {
				registeredStreamFn = config.streamSimple;
			},
		};
		const { default: factory } = await import("../index.js");
		factory(mockPi);
		if (!registeredStreamFn) throw new Error("no streamSimple");

		const model = fakeModel();
		const systemPrompt = "You are a helpful assistant. " + "x".repeat(2048); // ensure caching kicks in
		const messages: any[] = [{ role: "user", content: "What is 1+1?", timestamp: Date.now() }];

		const runOnce = async () => {
			const stream = (registeredStreamFn as Function)(model, { systemPrompt, messages }, {
				apiKey: process.env.ANTHROPIC_API_KEY,
			});
			let msg: any = null;
			for await (const event of stream) {
				if (event.type === "done") msg = event.message;
			}
			return msg;
		};

		const first = await runOnce();
		expect(first.stopReason).not.toBe("error");
		// First call should write to cache
		expect(first.usage.cacheWrite).toBeGreaterThan(0);

		const second = await runOnce();
		expect(second.stopReason).not.toBe("error");
		// Second call should read from cache
		expect(second.usage.cacheRead).toBeGreaterThan(0);
	});
});
