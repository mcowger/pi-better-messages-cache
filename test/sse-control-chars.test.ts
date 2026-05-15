/**
 * Tests for the SSE control-character fix (PR #4).
 *
 * The Anthropic SDK's stream() uses bare JSON.parse on each SSE data payload.
 * When the model emits raw control characters (\t, \n, etc.) inside a
 * tool-call JSON string (e.g. tab-indented code in an Edit tool call),
 * the wire SSE event contains LITERAL 0x09 / 0x0A bytes inside the JSON
 * string value:
 *
 *   data: {"delta":{"type":"input_json_delta","partial_json":"<TAB><TAB>env: {"}}
 *                                         ^^^^^^^^^^^ literal tab bytes
 *
 * JSON.parse throws:
 *   "Bad control character in string literal in JSON at position N"
 *
 * This propagates out of the for-await loop, cuts the stream before
 * content_block_stop fires, and leaves the tool-call arguments as {}.
 *
 * The fix replaces SDK stream() with:
 *   1. iterateSseMessages  — raw SSE frame parser
 *   2. iterateAnthropicSseEvents — uses parseJsonWithRepair instead of JSON.parse
 *
 * parseJsonWithRepair escapes raw control chars inside JSON string literals
 * before handing off to JSON.parse, so the data round-trips correctly.
 *
 * IMPORTANT: In these tests we must construct SSE wire data with ACTUAL raw
 * control-character bytes (0x09, 0x0A, etc.) inside JSON string values —
 * NOT the JSON-escaped forms (\t → \\t).  JSON.stringify escapes tabs to
 * \\t in its output, so we cannot use it to build the problematic payloads.
 * Instead we manually construct the JSON strings with real tab/newline bytes
 * embedded, exactly as Anthropic sends them.
 */

import { describe, expect, it } from "vitest";
import {
	iterateSseMessages,
	iterateAnthropicSseEvents,
} from "../index.js";
import { parseJsonWithRepair, parseStreamingJson } from "@earendil-works/pi-ai";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a ReadableStream<Uint8Array> from a raw SSE text string. */
function sseStream(text: string): ReadableStream<Uint8Array> {
	const encoder = new TextEncoder();
	return new ReadableStream({
		start(controller) {
			controller.enqueue(encoder.encode(text));
			controller.close();
		},
	});
}

/** Collect all events from an async generator into an array. */
async function collect<T>(gen: AsyncGenerator<T>): Promise<T[]> {
	const items: T[] = [];
	for await (const item of gen) {
		items.push(item);
	}
	return items;
}

/** Create a Response object wrapping the given SSE text. */
function sseResponse(text: string, headers: Record<string, string> = {}): Response {
	return new Response(sseStream(text), {
		status: 200,
		headers: { "content-type": "text/event-stream", ...headers },
	});
}

/**
 * Build an SSE data line whose JSON payload contains a raw control character
 * inside a string value — the exact condition that breaks JSON.parse.
 *
 * We construct the JSON manually (not via JSON.stringify) so that the
 * control character appears as a raw byte, e.g.:
 *   {"partial_json":"<TAB><TAB>env: {"}
 * not:
 *   {"partial_json":"\\t\\tenv: {"}
 *
 * @param before  JSON text before the value that will contain the control char
 * @param char    The raw control character to embed (e.g. "\t", "\n")
 * @param after   JSON text after the control char, closing the string + JSON
 */
function sseDataWithRawControlChar(
	eventType: string,
	before: string,
	char: string,
	after: string,
): string {
	return `event: ${eventType}\ndata: ${before}${char}${after}\n\n`;
}

// ---------------------------------------------------------------------------
// 1. parseJsonWithRepair vs JSON.parse — the core problem
// ---------------------------------------------------------------------------

describe("parseJsonWithRepair — control character handling", () => {
	it("JSON.parse FAILS on raw tabs inside a JSON string value (the bug)", () => {
		// This is exactly what Anthropic sends on the wire when the model emits
		// tab-indented code inside a tool-call argument. The tab is a literal
		// 0x09 byte, NOT the two-character escape \\t.
		const raw = '{"partial_json":"\t\tenv: {"}';

		expect(() => JSON.parse(raw)).toThrow(
			"Bad control character in string literal in JSON",
		);
	});

	it("parseJsonWithRepair SUCCEEDS on the same data (the fix)", () => {
		const raw = '{"partial_json":"\t\tenv: {"}';

		const result = parseJsonWithRepair(raw) as any;
		expect(result.partial_json).toBe("\t\tenv: {");
	});

	it("handles raw newlines inside JSON string values", () => {
		// Literal 0x0A byte inside a JSON string
		const raw = '{"text":"line1\nline2\nline3"}';

		expect(() => JSON.parse(raw)).toThrow();

		const result = parseJsonWithRepair(raw) as any;
		expect(result.text).toBe("line1\nline2\nline3");
	});

	it("handles mixed control characters (tab, newline, carriage return)", () => {
		// Raw \t, \n, \r bytes
		const raw = '{"code":"\t\tdef foo():\n\t\t\treturn 42\r\n"}';

		expect(() => JSON.parse(raw)).toThrow();

		const result = parseJsonWithRepair(raw) as any;
		expect(result.code).toBe("\t\tdef foo():\n\t\t\treturn 42\r\n");
	});

	it("preserves properly escaped control characters unchanged", () => {
		// Properly escaped \\t and \\n — JSON.stringify output style.
		// Both parsers should handle this identically.
		const properlyEscaped = '{"text":"hello\\nworld\\t!"}';
		expect(JSON.parse(properlyEscaped)).toEqual(parseJsonWithRepair(properlyEscaped));
	});
});

// ---------------------------------------------------------------------------
// 2. parseStreamingJson — handles partial JSON during streaming
// ---------------------------------------------------------------------------

describe("parseStreamingJson — control characters in partial tool-call JSON", () => {
	it("returns valid object even when partial JSON contains raw tabs", () => {
		// Simulates an in-flight input_json_delta with raw tabs
		const partial = '{"command":"\t\techo hello"}';
		const result = parseStreamingJson(partial);
		expect(result).toEqual({ command: "\t\techo hello" });
	});

	it("returns empty object for empty/trivially incomplete JSON", () => {
		expect(parseStreamingJson("")).toEqual({});
		expect(parseStreamingJson(" ")).toEqual({});
	});

	it("can partially parse an incomplete key-value pair with control chars", () => {
		// partial_json might be something like '{"oldText":"\t\tso'  (incomplete)
		const partial = '{"oldText":"\t\tso';
		const result = parseStreamingJson(partial);
		// Should return something (possibly partial), not throw
		expect(typeof result).toBe("object");
	});
});

// ---------------------------------------------------------------------------
// 3. iterateSseMessages — raw SSE frame parsing
// ---------------------------------------------------------------------------

describe("iterateSseMessages — SSE frame parsing", () => {
	it("parses a single SSE event", async () => {
		const stream = sseStream("event: ping\ndata: {}\n\n");
		const events = await collect(iterateSseMessages(stream));

		expect(events).toHaveLength(1);
		expect(events[0].event).toBe("ping");
		expect(events[0].data).toBe("{}");
	});

	it("parses multiple SSE events", async () => {
		const raw = [
			"event: message_start",
			'data: {"type":"message_start"}',
			"",
			"event: content_block_start",
			'data: {"type":"content_block_start"}',
			"",
			"",
		].join("\n");

		const events = await collect(iterateSseMessages(sseStream(raw)));
		expect(events).toHaveLength(2);
		expect(events[0].event).toBe("message_start");
		expect(events[1].event).toBe("content_block_start");
	});

	it("handles multi-line data fields (joins with newline)", async () => {
		const raw = "event: test\ndata: line1\ndata: line2\n\n";
		const events = await collect(iterateSseMessages(sseStream(raw)));

		expect(events).toHaveLength(1);
		expect(events[0].data).toBe("line1\nline2");
	});

	it("handles events with no event type (data-only)", async () => {
		const raw = 'data: {"hello":"world"}\n\n';
		const events = await collect(iterateSseMessages(sseStream(raw)));

		expect(events).toHaveLength(1);
		expect(events[0].event).toBeNull();
		expect(events[0].data).toBe('{"hello":"world"}');
	});

	it("preserves raw control characters in the data payload unmodified", async () => {
		// The data line contains a literal tab byte (0x09) inside the JSON.
		// iterateSseMessages should pass the raw data through; it's
		// iterateAnthropicSseEvents + parseJsonWithRepair that handle the repair.
		//
		// We build this manually: {"partial_json":"<TAB><TAB>env: {"}
		// with real tab bytes embedded.
		const dataPayload = '{"partial_json":"' + "\t\tenv: {" + '"}';
		const raw = `event: content_block_delta\ndata: ${dataPayload}\n\n`;
		const events = await collect(iterateSseMessages(sseStream(raw)));

		expect(events).toHaveLength(1);
		// The data should contain the raw tab characters exactly as received
		expect(events[0].data).toBe(dataPayload);
		expect(events[0].data).toContain("\t");
	});
});

// ---------------------------------------------------------------------------
// 4. iterateAnthropicSseEvents — end-to-end SSE with control chars
// ---------------------------------------------------------------------------

describe("iterateAnthropicSseEvents — control character handling", () => {
	it("successfully parses a tool-call delta containing raw tab characters", async () => {
		// Build SSE data with LITERAL tab bytes in the partial_json value.
		// This is the exact scenario from the bug report: Anthropic sends
		// raw \t (0x09) inside an input_json_delta's partial_json.
		//
		// Wire data looks like:
		//   data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"<TAB><TAB>env: {"}}
		//
		// With real tab bytes where <TAB> is shown.
		const partialJsonWithTabs = "\t\tenv: {";
		const dataPayload =
			'{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"' +
			partialJsonWithTabs +
			'"}}';
		const raw = `event: content_block_delta\ndata: ${dataPayload}\n\n`;

		const response = sseResponse(raw);
		const events = await collect(iterateAnthropicSseEvents(response));

		expect(events).toHaveLength(1);
		expect(events[0].type).toBe("content_block_delta");
		expect(events[0].delta.partial_json).toBe("\t\tenv: {");
	});

	it("successfully parses a tool-call delta containing raw tab + backspace characters", async () => {
		// Backspace (0x08) is another control char that survives SSE framing
		// (it's not a line separator) but breaks JSON.parse.
		const partialJsonWithCtrl = "\t\tdata\b";
		const dataPayload =
			'{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"' +
			partialJsonWithCtrl +
			'"}}';
		const raw = `event: content_block_delta\ndata: ${dataPayload}\n\n`;

		const response = sseResponse(raw);
		const events = await collect(iterateAnthropicSseEvents(response));

		expect(events).toHaveLength(1);
		expect(events[0].delta.partial_json).toBe("\t\tdata\b");
	});

	it("would FAIL with JSON.parse on raw tab data (proving the bug exists)", () => {
		// Direct proof that the old code path (JSON.parse) would fail:
		// This is the data exactly as iterateSseMessages would pass it
		// to iterateAnthropicSseEvents, and what the old code would try to parse.
		const partialJsonWithTabs = "\t\tenv: {";
		const dataPayload =
			'{"type":"content_block_delta","delta":{"partial_json":"' +
			partialJsonWithTabs +
			'"}}';

		// If iterateAnthropicSseEvents used JSON.parse (the old approach),
		// this would throw. Confirm:
		expect(() => JSON.parse(dataPayload)).toThrow(
			"Bad control character in string literal in JSON",
		);

		// But parseJsonWithRepair (the new approach) succeeds:
		const result = parseJsonWithRepair(dataPayload) as any;
		expect(result.delta.partial_json).toBe("\t\tenv: {");
	});

	it("skips non-Anthropic event types (e.g. comment events)", async () => {
		const raw = [
			"event: comment",
			'data: "some comment"',
			"",
			"event: message_start",
			'data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"model":"claude","usage":{"input_tokens":0,"output_tokens":0}}}',
			"",
		].join("\n");

		const response = sseResponse(raw);
		const events = await collect(iterateAnthropicSseEvents(response));

		// Only the message_start should be emitted; "comment" is not in ANTHROPIC_STREAM_EVENTS
		expect(events).toHaveLength(1);
		expect(events[0].type).toBe("message_start");
	});

	it("throws on SSE error events", async () => {
		const raw = "event: error\ndata: something went wrong\n\n";
		const response = sseResponse(raw);

		await expect(collect(iterateAnthropicSseEvents(response))).rejects.toThrow(
			"something went wrong",
		);
	});

	it("throws on malformed JSON with a descriptive error", async () => {
		const raw = 'event: message_start\ndata: {not json at all!!\n\n';
		const response = sseResponse(raw);

		await expect(collect(iterateAnthropicSseEvents(response))).rejects.toThrow(
			/Could not parse Anthropic SSE event/,
		);
	});

	it("handles a full multi-event stream with control-char tool-call JSON", async () => {
		// Simulates a realistic stream with a tool-call that has tab-indented
		// content — the exact scenario that caused the original bug.
		//
		// Build the input_json_delta data with REAL tab bytes in partial_json.
		// partial_json is a fragment of the tool-call JSON being streamed,
		// so early deltas might just be partial property values with raw tabs.
		const partialJsonWithTabs = "\t\tenv: config\t\tport: 8080";
		const jsonDeltaData =
			'{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"' +
			partialJsonWithTabs +
			'"}}';

		const raw = [
			// message_start
			'event: message_start',
			'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet","usage":{"input_tokens":100,"output_tokens":0}}}',
			'',
			// content_block_start: tool_use
			'event: content_block_start',
			'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"Edit","input":{}}}',
			'',
			// content_block_delta: input_json_delta with RAW TABS
			'event: content_block_delta',
			`data: ${jsonDeltaData}`,
			'',
			// content_block_stop
			'event: content_block_stop',
			'data: {"type":"content_block_stop","index":0}',
			'',
			// message_delta (stop reason + usage)
			'event: message_delta',
			'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":50}}',
			'',
			// message_stop
			'event: message_stop',
			'data: {"type":"message_stop"}',
			'',
		].join('\n');

		const response = sseResponse(raw);
		const events = await collect(iterateAnthropicSseEvents(response));

		// All 6 events should parse successfully — no crash from control chars
		const types = events.map((e: any) => e.type);
		expect(types).toEqual([
			"message_start",
			"content_block_start",
			"content_block_delta",
			"content_block_stop",
			"message_delta",
			"message_stop",
		]);

		// The critical part: the tool-call delta's partial_json contains real tabs
		const delta = events.find((e: any) => e.type === "content_block_delta");
		expect(delta.delta.partial_json).toContain("\t");
	});
});
