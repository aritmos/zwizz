const std = @import("std");
const swiss = @import("swiss_hash_map.zig");
const mock = @import("mock_swiss_hash_map.zig");
const hm = @import("hash_map.zig");

pub fn main() !void {
    run(std.hash_map.AutoHashMap(u32, u32));
    // run(swiss.AutoSwissHashMap(u32, u32));
    // const l = lookups(mock.AutoHashMap(u32, u32));
    // std.debug.print("{}\n", .{l});
}

// 10M random put and remove
fn run(Map: type) void {
    const n = 10_000 * 1000;
    var map = Map.init(std.heap.page_allocator);
    defer map.deinit();

    var keys = std.ArrayList(u32).init(std.heap.page_allocator);
    defer keys.deinit();

    var i: u32 = 0;
    while (i < n) : (i += 1) {
        keys.append(i) catch unreachable;
    }

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();
    random.shuffle(u32, keys.items);

    for (keys.items) |key| {
        map.put(key, key) catch unreachable;
    }

    random.shuffle(u32, keys.items);
    i = 0;
    while (i < n) : (i += 1) {
        const key = keys.items[i];
        _ = map.remove(key);
    }
}
