import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, initialBase, normaliseBase } from "./api";
import { duration, ms, pct } from "./format";

const store: Record<string, string> = {};
beforeEach(() => {
  vi.stubGlobal("window", {
    localStorage: {
      getItem: (k: string) => store[k] ?? null,
      setItem: (k: string, v: string) => void (store[k] = v),
    },
    location: { protocol: "http:", hostname: "ui-host" },
  });
});
afterEach(() => {
  vi.unstubAllGlobals();
  for (const k of Object.keys(store)) delete store[k];
});

describe("API address", () => {
  it("normalises user input", () => {
    expect(normaliseBase(" 192.168.1.5:8000/ ")).toBe("http://192.168.1.5:8000");
    expect(normaliseBase("https://x.test//")).toBe("https://x.test");
  });
  it("prefers saved > env > same host", () => {
    expect(initialBase(undefined, { protocol: "http:", hostname: "ui-host" })).toBe("http://ui-host:8000");
    expect(initialBase("10.0.0.2:8000", { protocol: "http:", hostname: "ui-host" })).toBe("http://10.0.0.2:8000");
    store["pathsnap.apiBase"] = "http://saved:8000";
    expect(initialBase("10.0.0.2:8000")).toBe("http://saved:8000");
  });
});

describe("request errors", () => {
  it("maps API error bodies", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(JSON.stringify({ detail: "Unknown trip 'x'", code: "TRIP_NOT_FOUND" }), { status: 404 })));
    await expect(api.predict("http://a", "x")).rejects.toMatchObject({ status: 404, code: "TRIP_NOT_FOUND" });
  });
  it("reports unreachable hosts", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const err = await api.health("http://down:8000").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("UNREACHABLE");
  });
});

describe("formatting", () => {
  it("formats numbers for the result panel", () => {
    expect(pct(0.92187)).toBe("92.2%");
    expect(pct(null)).toBe("n/a");
    expect(duration(3725)).toBe("1 h 02 min");
    expect(ms(412.3)).toBe("412 ms");
    expect(ms(1617)).toBe("1.62 s");
  });
});
