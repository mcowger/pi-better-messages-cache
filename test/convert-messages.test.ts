/**
 * Unit tests for convertMessages() — the core of the dual cache-breakpoint logic.
 *
 * All tests run without a network connection or API key.
 *
 * Conventions
 * -----------
 * - Messages are constructed as plain objects and cast to `any` so tests stay
 *   self-contained without importing the full pi-ai type tree.
 * - `CC` == cache_control: { type: "ephemeral" }
 * - "dual-cache" == both the last assistant tool_use block AND the last user
 *   message block carry CC (the feature this package adds).
 */

import { describe, expect, it } from "vitest";
import { convertMessages } from "../index.js";

// Shorthand for the cache_control value placed by this extension
const CC = { type: "ephemeral" };
const TS = 0; // timestamp placeholder

// ---------------------------------------------------------------------------
// Helpers to build message objects
// ---------------------------------------------------------------------------

function userMsg(content: string | any[]): any {
	return { role: "user", content, timestamp: TS };
}

function assistantMsg(content: any[]): any {
	return {
		role: "assistant",
		content,
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4-20250514",
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
		stopReason: "toolUse",
		timestamp: TS,
	};
}

function toolResultMsg(id: string, text: string, isError = false): any {
	return {
		role: "toolResult",
		toolCallId: id,
		toolName: "bash",
		content: [{ type: "text", text }],
		isError,
		timestamp: TS,
	};
}

function textBlock(text: string): any {
	return { type: "text", text };
}

function thinkingBlock(thinking: string, sig = "sig123"): any {
	return { type: "thinking", thinking, thinkingSignature: sig };
}

function toolCallBlock(id: string, name: string, args: Record<string, any> = {}): any {
	return { type: "toolCall", id, name, arguments: args };
}

const CACHE = CC; // alias

// ---------------------------------------------------------------------------
// 1. User messages
// ---------------------------------------------------------------------------

describe("user messages", () => {
	it("promotes string content to array block with cache_control", () => {
		const msgs = [userMsg("hello")];
		const result = convertMessages(msgs, false, CC);

		expect(result).toHaveLength(1);
		expect(result[0].role).toBe("user");
		// String content is promoted to [{type:'text', text:'hello', cache_control}]
		expect(Array.isArray(result[0].content)).toBe(true);
		expect(result[0].content[0]).toEqual({ type: "text", text: "hello", cache_control: CACHE });
	});

	it("places cache_control on the last block of an array user message", () => {
		const msgs = [
			userMsg([
				{ type: "text", text: "first" },
				{ type: "text", text: "second" },
			]),
		];
		const result = convertMessages(msgs, false, CC);

		const content = result[0].content;
		expect(content[0].cache_control).toBeUndefined();
		expect(content[1].cache_control).toEqual(CACHE);
	});

	it("only marks the LAST user message, not earlier ones", () => {
		const msgs = [
			userMsg("turn one"),       // not last user message
			assistantMsg([textBlock("ok")]),
			userMsg("turn two"),       // last user message → gets CC
		];
		const result = convertMessages(msgs, false, CC);

		// find all user messages in result
		const userMsgs = result.filter((m: any) => m.role === "user");
		expect(userMsgs).toHaveLength(2);

		// first user message: no cache_control
		const firstContent = userMsgs[0].content;
		if (Array.isArray(firstContent)) {
			const last = firstContent[firstContent.length - 1];
			expect(last.cache_control).toBeUndefined();
		}

		// last user message: has cache_control
		const lastContent = userMsgs[1].content;
		const lastBlock = Array.isArray(lastContent)
			? lastContent[lastContent.length - 1]
			: null;
		if (lastBlock) {
			expect(lastBlock.cache_control).toEqual(CACHE);
		} else {
			// String content promoted to array
			expect(result[result.length - 1].content[0].cache_control).toEqual(CACHE);
		}
	});

	it("filters out purely whitespace text blocks", () => {
		const msgs = [userMsg([{ type: "text", text: "   " }])];
		const result = convertMessages(msgs, false, CC);
		// whitespace-only block should be filtered → no message produced
		expect(result).toHaveLength(0);
	});

	it("skips empty string content", () => {
		const msgs = [userMsg("")];
		const result = convertMessages(msgs, false, CC);
		expect(result).toHaveLength(0);
	});
});

// ---------------------------------------------------------------------------
// 2. Assistant messages — cache_control on the last tool_use block
// ---------------------------------------------------------------------------

describe("assistant messages — tool_use cache_control (dual-cache change 1 of 2)", () => {
	it("marks a single tool_use block with cache_control", () => {
		const msgs = [
			assistantMsg([toolCallBlock("id1", "bash", { command: "ls" })]),
			toolResultMsg("id1", "file.txt"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		expect(assistantResult).toBeDefined();
		const toolUse = assistantResult!.content[0];
		expect(toolUse.type).toBe("tool_use");
		expect(toolUse.cache_control).toEqual(CACHE);
	});

	it("marks only the LAST tool_use block when there are multiple", () => {
		const msgs = [
			assistantMsg([
				toolCallBlock("id1", "read"),
				toolCallBlock("id2", "bash"),   // ← last → gets CC
			]),
			toolResultMsg("id1", "content"),
			toolResultMsg("id2", "output"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		const [first, second] = assistantResult!.content;

		expect(first.cache_control).toBeUndefined();   // not last
		expect(second.cache_control).toEqual(CACHE);   // last
	});

	it("does NOT add cache_control to assistant tool_use when there are no tool calls (text only)", () => {
		const msgs = [assistantMsg([textBlock("Just some text.")])];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		expect(assistantResult!.content[0].cache_control).toBeUndefined();
	});

	it("does NOT add cache_control to text blocks even when tool calls are present", () => {
		const msgs = [
			assistantMsg([
				textBlock("I will run this."),
				toolCallBlock("id1", "bash"),
			]),
			toolResultMsg("id1", "done"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		const textEntry = assistantResult!.content.find((b: any) => b.type === "text");
		const toolUseEntry = assistantResult!.content.find((b: any) => b.type === "tool_use");

		expect(textEntry!.cache_control).toBeUndefined();
		expect(toolUseEntry!.cache_control).toEqual(CACHE);
	});

	it("marks the last tool_use even when thinking blocks precede it", () => {
		const msgs = [
			assistantMsg([
				thinkingBlock("Let me think..."),
				toolCallBlock("id1", "bash"),
			]),
			toolResultMsg("id1", "result"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		const thinkingEntry = assistantResult!.content.find((b: any) => b.type === "thinking");
		const toolUseEntry = assistantResult!.content.find((b: any) => b.type === "tool_use");

		expect(thinkingEntry!.cache_control).toBeUndefined();
		expect(toolUseEntry!.cache_control).toEqual(CACHE);
	});

	it("marks the last tool_use when multiple thinking blocks and tool calls are interleaved", () => {
		const msgs = [
			assistantMsg([
				thinkingBlock("think 1"),
				toolCallBlock("id1", "read"),
				thinkingBlock("think 2"),
				toolCallBlock("id2", "write"),  // ← last tool call → gets CC
			]),
			toolResultMsg("id1", "r1"),
			toolResultMsg("id2", "r2"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		const toolUses = assistantResult!.content.filter((b: any) => b.type === "tool_use");

		expect(toolUses).toHaveLength(2);
		expect(toolUses[0].cache_control).toBeUndefined();
		expect(toolUses[1].cache_control).toEqual(CACHE);
	});

	it("filters out empty text blocks from assistant messages", () => {
		const msgs = [
			assistantMsg([
				textBlock(""),         // empty → filtered
				toolCallBlock("id1", "bash"),
			]),
			toolResultMsg("id1", "done"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		const allTypes = assistantResult!.content.map((b: any) => b.type);
		expect(allTypes).not.toContain("text");
		expect(allTypes).toContain("tool_use");
	});

	it("filters out empty thinking blocks from assistant messages", () => {
		const msgs = [
			assistantMsg([
				thinkingBlock(""),       // empty → filtered
				toolCallBlock("id1", "bash"),
			]),
			toolResultMsg("id1", "done"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		const allTypes = assistantResult!.content.map((b: any) => b.type);
		expect(allTypes).not.toContain("thinking");
	});

	it("converts thinking blocks that have no thinkingSignature to plain text blocks", () => {
		const msgs = [
			assistantMsg([
				{ type: "thinking", thinking: "some thought", thinkingSignature: "" },
			]),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		expect(assistantResult!.content[0].type).toBe("text");
		expect(assistantResult!.content[0].text).toBe("some thought");
	});

	it("passes through redacted thinking blocks as redacted_thinking", () => {
		const msgs = [
			assistantMsg([
				{ type: "thinking", thinking: "", redacted: true, thinkingSignature: "opaque-payload" },
			]),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		expect(assistantResult!.content[0].type).toBe("redacted_thinking");
		expect(assistantResult!.content[0].data).toBe("opaque-payload");
	});

	it("produces no assistant message when all content blocks are empty", () => {
		const msgs = [
			assistantMsg([textBlock(""), thinkingBlock("")]),
		];
		const result = convertMessages(msgs, false, CC);
		const assistantEntries = result.filter((m: any) => m.role === "assistant");
		expect(assistantEntries).toHaveLength(0);
	});
});

// ---------------------------------------------------------------------------
// 3. Tool result messages
// ---------------------------------------------------------------------------

describe("tool result messages", () => {
	it("converts a single tool result into a user message with tool_result block", () => {
		const msgs = [
			assistantMsg([toolCallBlock("id1", "bash")]),
			toolResultMsg("id1", "hello"),
		];
		const result = convertMessages(msgs, false, CC);

		const userMessages = result.filter((m: any) => m.role === "user");
		expect(userMessages).toHaveLength(1);
		const [tr] = userMessages[0].content;
		expect(tr.type).toBe("tool_result");
		expect(tr.tool_use_id).toBe("id1");
	});

	it("batches consecutive tool results into a single user message", () => {
		const msgs = [
			assistantMsg([
				toolCallBlock("id1", "read"),
				toolCallBlock("id2", "bash"),
			]),
			toolResultMsg("id1", "file content"),
			toolResultMsg("id2", "exit 0"),
		];
		const result = convertMessages(msgs, false, CC);

		const userMessages = result.filter((m: any) => m.role === "user");
		// Only one batched user message for the two consecutive toolResults
		expect(userMessages).toHaveLength(1);
		expect(userMessages[0].content).toHaveLength(2);
		expect(userMessages[0].content[0].tool_use_id).toBe("id1");
		expect(userMessages[0].content[1].tool_use_id).toBe("id2");
	});

	it("places cache_control on the last tool_result block (final user message)", () => {
		const msgs = [
			assistantMsg([toolCallBlock("id1", "bash")]),
			toolResultMsg("id1", "done"),
		];
		const result = convertMessages(msgs, false, CC);

		const lastUser = result[result.length - 1];
		const lastBlock = lastUser.content[lastUser.content.length - 1];
		expect(lastBlock.cache_control).toEqual(CACHE);
	});

	it("marks tool_result as error when isError is true", () => {
		const msgs = [
			assistantMsg([toolCallBlock("id1", "bash")]),
			toolResultMsg("id1", "command not found", true),
		];
		const result = convertMessages(msgs, false, CC);

		const lastUser = result[result.length - 1];
		expect(lastUser.content[0].is_error).toBe(true);
	});
});

// ---------------------------------------------------------------------------
// 4. Multi-turn conversations
// ---------------------------------------------------------------------------

describe("multi-turn conversations", () => {
	it("marks the last tool_use in EACH assistant message (both turns get CC)", () => {
		const msgs = [
			// Turn 1
			userMsg("do the thing"),
			assistantMsg([toolCallBlock("id1", "bash")]),
			toolResultMsg("id1", "done step 1"),
			// Turn 2
			assistantMsg([toolCallBlock("id2", "write")]),
			toolResultMsg("id2", "done step 2"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantMsgs = result.filter((m: any) => m.role === "assistant");
		expect(assistantMsgs).toHaveLength(2);

		// Both assistant messages should have their last tool_use marked
		for (const am of assistantMsgs) {
			const toolUses = am.content.filter((b: any) => b.type === "tool_use");
			const lastToolUse = toolUses[toolUses.length - 1];
			expect(lastToolUse.cache_control).toEqual(CACHE);
		}
	});

	it("only the FINAL user/tool-result message gets cache_control (not earlier ones)", () => {
		const msgs = [
			userMsg("start"),
			assistantMsg([toolCallBlock("id1", "bash")]),
			toolResultMsg("id1", "result1"),          // NOT last user msg
			assistantMsg([toolCallBlock("id2", "bash")]),
			toolResultMsg("id2", "result2"),          // last user msg → gets CC
		];
		const result = convertMessages(msgs, false, CC);

		const userMsgs = result.filter((m: any) => m.role === "user");
		expect(userMsgs.length).toBeGreaterThanOrEqual(3);

		// All but the last user message should NOT have cache_control on their last block
		for (const um of userMsgs.slice(0, -1)) {
			const lastBlock = um.content[um.content.length - 1];
			expect(lastBlock.cache_control).toBeUndefined();
		}

		// Last user message SHOULD have cache_control
		const lastUM = userMsgs[userMsgs.length - 1];
		const lastBlock = lastUM.content[lastUM.content.length - 1];
		expect(lastBlock.cache_control).toEqual(CACHE);
	});

	it("preserves message order across multiple turns", () => {
		const msgs = [
			userMsg("user 1"),
			assistantMsg([textBlock("assistant 1")]),
			userMsg("user 2"),
			assistantMsg([textBlock("assistant 2")]),
		];
		const result = convertMessages(msgs, false, CC);

		expect(result[0].role).toBe("user");
		expect(result[1].role).toBe("assistant");
		expect(result[2].role).toBe("user");
		expect(result[3].role).toBe("assistant");
	});
});

// ---------------------------------------------------------------------------
// 5. OAuth / Claude Code tool name normalisation
// ---------------------------------------------------------------------------

describe("OAuth mode — Claude Code tool name normalisation", () => {
	it("converts recognised tool names to PascalCase when isOAuth=true", () => {
		const msgs = [
			assistantMsg([toolCallBlock("id1", "read")]),
			toolResultMsg("id1", "content"),
		];
		const result = convertMessages(msgs, true, CC);   // isOAuth = true

		const assistantResult = result.find((m: any) => m.role === "assistant");
		expect(assistantResult!.content[0].name).toBe("Read");
	});

	it("passes through unrecognised tool names unchanged in OAuth mode", () => {
		const msgs = [
			assistantMsg([toolCallBlock("id1", "my_custom_tool")]),
			toolResultMsg("id1", "result"),
		];
		const result = convertMessages(msgs, true, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		expect(assistantResult!.content[0].name).toBe("my_custom_tool");
	});

	it("keeps original tool names when isOAuth=false", () => {
		const msgs = [
			assistantMsg([toolCallBlock("id1", "bash")]),
			toolResultMsg("id1", "output"),
		];
		const result = convertMessages(msgs, false, CC);

		const assistantResult = result.find((m: any) => m.role === "assistant");
		expect(assistantResult!.content[0].name).toBe("bash");
	});
});

// ---------------------------------------------------------------------------
// 6. Surrogate character sanitisation
// ---------------------------------------------------------------------------

describe("surrogate character sanitisation", () => {
	it("replaces lone surrogates in user text with replacement char", () => {
		const msgs = [userMsg([{ type: "text", text: "bad \uD800 char" }])];
		const result = convertMessages(msgs, false, CC);
		expect(result[0].content[0].text).toBe("bad \uFFFD char");
	});

	it("replaces lone surrogates in assistant text blocks", () => {
		const msgs = [
			assistantMsg([textBlock("oops \uDFFF here")]),
		];
		const result = convertMessages(msgs, false, CC);
		const assistantResult = result.find((m: any) => m.role === "assistant");
		expect(assistantResult!.content[0].text).toBe("oops \uFFFD here");
	});
});

// ---------------------------------------------------------------------------
// 7. Edge cases
// ---------------------------------------------------------------------------

describe("edge cases", () => {
	it("returns an empty array for an empty message list", () => {
		expect(convertMessages([], false, CC)).toEqual([]);
	});

	it("handles a single user message with no prior context", () => {
		const msgs = [userMsg("hello")];
		const result = convertMessages(msgs, false, CC);
		expect(result).toHaveLength(1);
	});

	it("does not mutate the original messages array", () => {
		const original: any[] = [userMsg("test")];
		const copy = JSON.parse(JSON.stringify(original));
		convertMessages(original, false, CC);
		expect(original).toEqual(copy);
	});

	it("handles assistant message that appears as the final message (no trailing user)", () => {
		// Unusual but valid: conversation ends with an assistant text message.
		// No user message means no cache_control from part 2, but assistant
		// still gets its tool_use (if any) marked.
		const msgs = [
			userMsg("go"),
			assistantMsg([textBlock("done")]),
		];
		const result = convertMessages(msgs, false, CC);
		// Last message is assistant — no user-message CC applied, no error
		expect(result[result.length - 1].role).toBe("assistant");
	});
});
