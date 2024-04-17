const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const autoHash = std.hash.autoHash;
const math = std.math;
const mem = std.mem;
const Allocator = mem.Allocator;
const Wyhash = std.hash.Wyhash;

// Reuse functions and types from std/hash_map.zig
const hash_map = std.hash_map;
pub const getAutoHashFn = hash_map.getAutoHashFn;
pub const getAutoEqlFn = hash_map.getAutoEqlFn;
pub const AutoContext = hash_map.AutoContext;
pub const StringContext = hash_map.StringContext;
pub const eqlString = hash_map.eqlString;
pub const hashString = hash_map.hashString;
pub const StringIndexContext = hash_map.StringIndexContext;
pub const StringIndexAdapter = hash_map.StringIndexAdapter;
pub const verifyContext = hash_map.verifyContext;

pub fn AutoSwissHashMap(comptime K: type, comptime V: type) type {
    return SwissHashMap(K, V, AutoContext(K), default_max_load_percentage);
}

pub fn AutoSwissHashMapUnmanaged(comptime K: type, comptime V: type) type {
    return SwissHashMapUnmanaged(K, V, AutoContext(K), default_max_load_percentage);
}

/// Builtin hashmap for strings as keys.
/// Key memory is managed by the caller.  Keys and values
/// will not automatically be freed.
pub fn StringSwissHashMap(comptime V: type) type {
    return SwissHashMap([]const u8, V, StringContext, default_max_load_percentage);
}

/// Key memory is managed by the caller.  Keys and values
/// will not automatically be freed.
pub fn StringSwissHashMapUnmanaged(comptime V: type) type {
    return SwissHashMapUnmanaged([]const u8, V, StringContext, default_max_load_percentage);
}

pub const default_max_load_percentage = 80;

/// General purpose hash table.
/// No order is guaranteed and any modification invalidates live iterators.
/// It provides fast operations (lookup, insertion, deletion) with quite high
/// load factors (up to 80% by default) for low memory usage.
/// For a hash map that can be initialized directly that does not store an Allocator
/// field, see `HashMapUnmanaged`.
/// If iterating over the table entries is a strong usecase and needs to be fast,
/// prefer the alternative `std.ArrayHashMap`.
/// Context must be a struct type with two member functions:
///   hash(self, K) u64
///   eql(self, K, K) bool
/// Adapted variants of many functions are provided.  These variants
/// take a pseudo key instead of a key.  Their context must have the functions:
///   hash(self, PseudoKey) u64
///   eql(self, PseudoKey, K) bool
pub fn SwissHashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime max_load_percentage: u64,
) type {
    return struct {
        unmanaged: Unmanaged,
        allocator: Allocator,
        ctx: Context,

        comptime {
            verifyContext(Context, K, K, u64, false);
        }

        /// The type of the unmanaged hash map underlying this wrapper
        pub const Unmanaged = SwissHashMapUnmanaged(K, V, Context, max_load_percentage);
        /// An entry, containing pointers to a key and value stored in the map
        pub const Entry = Unmanaged.Entry;
        /// A copy of a key and value which are no longer in the map
        pub const KV = Unmanaged.KV;
        /// The integer type that is the result of hashing
        pub const Hash = Unmanaged.Hash;
        /// The iterator type returned by iterator()
        pub const Iterator = Unmanaged.Iterator;

        pub const KeyIterator = Unmanaged.KeyIterator;
        pub const ValueIterator = Unmanaged.ValueIterator;

        /// The integer type used to store the size of the map
        pub const Size = Unmanaged.Size;
        /// The type returned from getOrPut and variants
        pub const GetOrPutResult = Unmanaged.GetOrPutResult;

        const Self = @This();

        /// Create a managed hash map with an empty context.
        /// If the context is not zero-sized, you must use
        /// initContext(allocator, ctx) instead.
        pub fn init(allocator: Allocator) Self {
            if (@sizeOf(Context) != 0) {
                @compileError("Context must be specified! Call initContext(allocator, ctx) instead.");
            }
            return .{
                .unmanaged = .{},
                .allocator = allocator,
                .ctx = undefined, // ctx is zero-sized so this is safe.
            };
        }

        /// Create a managed hash map with a context
        pub fn initContext(allocator: Allocator, ctx: Context) Self {
            return .{
                .unmanaged = .{},
                .allocator = allocator,
                .ctx = ctx,
            };
        }

        /// Release the backing array and invalidate this map.
        /// This does *not* deinit keys, values, or the context!
        /// If your keys or values need to be released, ensure
        /// that that is done before calling this function.
        pub fn deinit(self: *Self) void {
            self.unmanaged.deinit(self.allocator);
            self.* = undefined;
        }

        /// Empty the map, but keep the backing allocation for future use.
        /// This does *not* free keys or values! Be sure to
        /// release them if they need deinitialization before
        /// calling this function.
        pub fn clearRetainingCapacity(self: *Self) void {
            return self.unmanaged.clearRetainingCapacity();
        }

        /// Empty the map and release the backing allocation.
        /// This does *not* free keys or values! Be sure to
        /// release them if they need deinitialization before
        /// calling this function.
        pub fn clearAndFree(self: *Self) void {
            return self.unmanaged.clearAndFree(self.allocator);
        }

        /// Return the number of items in the map.
        pub fn count(self: Self) Size {
            return self.unmanaged.count();
        }

        /// Create an iterator over the entries in the map.
        /// The iterator is invalidated if the map is modified.
        pub fn iterator(self: *const Self) Iterator {
            return self.unmanaged.iterator();
        }

        /// Create an iterator over the keys in the map.
        /// The iterator is invalidated if the map is modified.
        pub fn keyIterator(self: *const Self) KeyIterator {
            return self.unmanaged.keyIterator();
        }

        /// Create an iterator over the values in the map.
        /// The iterator is invalidated if the map is modified.
        pub fn valueIterator(self: *const Self) ValueIterator {
            return self.unmanaged.valueIterator();
        }

        /// If key exists this function cannot fail.
        /// If there is an existing item with `key`, then the result's
        /// `Entry` pointers point to it, and found_existing is true.
        /// Otherwise, puts a new item with undefined value, and
        /// the `Entry` pointers point to it. Caller should then initialize
        /// the value (but not the key).
        pub fn getOrPut(self: *Self, key: K) Allocator.Error!GetOrPutResult {
            return self.unmanaged.getOrPutContext(self.allocator, key, self.ctx);
        }

        /// If key exists this function cannot fail.
        /// If there is an existing item with `key`, then the result's
        /// `Entry` pointers point to it, and found_existing is true.
        /// Otherwise, puts a new item with undefined key and value, and
        /// the `Entry` pointers point to it. Caller must then initialize
        /// the key and value.
        pub fn getOrPutAdapted(self: *Self, key: anytype, ctx: anytype) Allocator.Error!GetOrPutResult {
            return self.unmanaged.getOrPutContextAdapted(self.allocator, key, ctx, self.ctx);
        }

        /// If there is an existing item with `key`, then the result's
        /// `Entry` pointers point to it, and found_existing is true.
        /// Otherwise, puts a new item with undefined value, and
        /// the `Entry` pointers point to it. Caller should then initialize
        /// the value (but not the key).
        /// If a new entry needs to be stored, this function asserts there
        /// is enough capacity to store it.
        pub fn getOrPutAssumeCapacity(self: *Self, key: K) GetOrPutResult {
            return self.unmanaged.getOrPutAssumeCapacityContext(key, self.ctx);
        }

        /// If there is an existing item with `key`, then the result's
        /// `Entry` pointers point to it, and found_existing is true.
        /// Otherwise, puts a new item with undefined value, and
        /// the `Entry` pointers point to it. Caller must then initialize
        /// the key and value.
        /// If a new entry needs to be stored, this function asserts there
        /// is enough capacity to store it.
        pub fn getOrPutAssumeCapacityAdapted(self: *Self, key: anytype, ctx: anytype) GetOrPutResult {
            return self.unmanaged.getOrPutAssumeCapacityAdapted(key, ctx);
        }

        pub fn getOrPutValue(self: *Self, key: K, value: V) Allocator.Error!Entry {
            return self.unmanaged.getOrPutValueContext(self.allocator, key, value, self.ctx);
        }

        /// Increases capacity, guaranteeing that insertions up until the
        /// `expected_count` will not cause an allocation, and therefore cannot fail.
        pub fn ensureTotalCapacity(self: *Self, expected_count: Size) Allocator.Error!void {
            return self.unmanaged.ensureTotalCapacityContext(self.allocator, expected_count, self.ctx);
        }

        /// Increases capacity, guaranteeing that insertions up until
        /// `additional_count` **more** items will not cause an allocation, and
        /// therefore cannot fail.
        pub fn ensureUnusedCapacity(self: *Self, additional_count: Size) Allocator.Error!void {
            return self.unmanaged.ensureUnusedCapacityContext(self.allocator, additional_count, self.ctx);
        }

        /// Returns the number of total elements which may be present before it is
        /// no longer guaranteed that no allocations will be performed.
        pub fn capacity(self: *Self) Size {
            return self.unmanaged.capacity();
        }

        /// Clobbers any existing data. To detect if a put would clobber
        /// existing data, see `getOrPut`.
        pub fn put(self: *Self, key: K, value: V) Allocator.Error!void {
            return self.unmanaged.putContext(self.allocator, key, value, self.ctx);
        }

        /// Inserts a key-value pair into the hash map, asserting that no previous
        /// entry with the same key is already present
        pub fn putNoClobber(self: *Self, key: K, value: V) Allocator.Error!void {
            return self.unmanaged.putNoClobberContext(self.allocator, key, value, self.ctx);
        }

        /// Asserts there is enough capacity to store the new key-value pair.
        /// Clobbers any existing data. To detect if a put would clobber
        /// existing data, see `getOrPutAssumeCapacity`.
        pub fn putAssumeCapacity(self: *Self, key: K, value: V) void {
            return self.unmanaged.putAssumeCapacityContext(key, value, self.ctx);
        }

        /// Asserts there is enough capacity to store the new key-value pair.
        /// Asserts that it does not clobber any existing data.
        /// To detect if a put would clobber existing data, see `getOrPutAssumeCapacity`.
        pub fn putAssumeCapacityNoClobber(self: *Self, key: K, value: V) void {
            return self.unmanaged.putAssumeCapacityNoClobberContext(key, value, self.ctx);
        }

        /// Inserts a new `Entry` into the hash map, returning the previous one, if any.
        pub fn fetchPut(self: *Self, key: K, value: V) Allocator.Error!?KV {
            return self.unmanaged.fetchPutContext(self.allocator, key, value, self.ctx);
        }

        /// Inserts a new `Entry` into the hash map, returning the previous one, if any.
        /// If insertion happens, asserts there is enough capacity without allocating.
        pub fn fetchPutAssumeCapacity(self: *Self, key: K, value: V) ?KV {
            return self.unmanaged.fetchPutAssumeCapacityContext(key, value, self.ctx);
        }

        /// Removes a value from the map and returns the removed kv pair.
        pub fn fetchRemove(self: *Self, key: K) ?KV {
            return self.unmanaged.fetchRemoveContext(key, self.ctx);
        }

        pub fn fetchRemoveAdapted(self: *Self, key: anytype, ctx: anytype) ?KV {
            return self.unmanaged.fetchRemoveAdapted(key, ctx);
        }

        /// Finds the value associated with a key in the map
        pub fn get(self: Self, key: K) ?V {
            return self.unmanaged.getContext(key, self.ctx);
        }
        pub fn getAdapted(self: Self, key: anytype, ctx: anytype) ?V {
            return self.unmanaged.getAdapted(key, ctx);
        }

        pub fn getPtr(self: Self, key: K) ?*V {
            return self.unmanaged.getPtrContext(key, self.ctx);
        }
        pub fn getPtrAdapted(self: Self, key: anytype, ctx: anytype) ?*V {
            return self.unmanaged.getPtrAdapted(key, ctx);
        }

        /// Finds the actual key associated with an adapted key in the map
        pub fn getKey(self: Self, key: K) ?K {
            return self.unmanaged.getKeyContext(key, self.ctx);
        }
        pub fn getKeyAdapted(self: Self, key: anytype, ctx: anytype) ?K {
            return self.unmanaged.getKeyAdapted(key, ctx);
        }

        pub fn getKeyPtr(self: Self, key: K) ?*K {
            return self.unmanaged.getKeyPtrContext(key, self.ctx);
        }
        pub fn getKeyPtrAdapted(self: Self, key: anytype, ctx: anytype) ?*K {
            return self.unmanaged.getKeyPtrAdapted(key, ctx);
        }

        /// Finds the key and value associated with a key in the map
        pub fn getEntry(self: Self, key: K) ?Entry {
            return self.unmanaged.getEntryContext(key, self.ctx);
        }

        pub fn getEntryAdapted(self: Self, key: anytype, ctx: anytype) ?Entry {
            return self.unmanaged.getEntryAdapted(key, ctx);
        }

        /// Check if the map contains a key
        pub fn contains(self: Self, key: K) bool {
            return self.unmanaged.containsContext(key, self.ctx);
        }

        pub fn containsAdapted(self: Self, key: anytype, ctx: anytype) bool {
            return self.unmanaged.containsAdapted(key, ctx);
        }

        /// If there is an `Entry` with a matching key, it is deleted from
        /// the hash map, and this function returns true.  Otherwise this
        /// function returns false.
        pub fn remove(self: *Self, key: K) bool {
            return self.unmanaged.removeContext(key, self.ctx);
        }

        pub fn removeAdapted(self: *Self, key: anytype, ctx: anytype) bool {
            return self.unmanaged.removeAdapted(key, ctx);
        }

        /// Delete the entry with key pointed to by key_ptr from the hash map.
        /// key_ptr is assumed to be a valid pointer to a key that is present
        /// in the hash map.
        pub fn removeByPtr(self: *Self, key_ptr: *K) void {
            self.unmanaged.removeByPtr(key_ptr);
        }

        /// Creates a copy of this map, using the same allocator
        pub fn clone(self: Self) Allocator.Error!Self {
            var other = try self.unmanaged.cloneContext(self.allocator, self.ctx);
            return other.promoteContext(self.allocator, self.ctx);
        }

        /// Creates a copy of this map, using a specified allocator
        pub fn cloneWithAllocator(self: Self, new_allocator: Allocator) Allocator.Error!Self {
            var other = try self.unmanaged.cloneContext(new_allocator, self.ctx);
            return other.promoteContext(new_allocator, self.ctx);
        }

        /// Creates a copy of this map, using a specified context
        pub fn cloneWithContext(self: Self, new_ctx: anytype) Allocator.Error!SwissHashMap(K, V, @TypeOf(new_ctx), max_load_percentage) {
            var other = try self.unmanaged.cloneContext(self.allocator, new_ctx);
            return other.promoteContext(self.allocator, new_ctx);
        }

        /// Creates a copy of this map, using a specified allocator and context.
        pub fn cloneWithAllocatorAndContext(
            self: Self,
            new_allocator: Allocator,
            new_ctx: anytype,
        ) Allocator.Error!SwissHashMap(K, V, @TypeOf(new_ctx), max_load_percentage) {
            var other = try self.unmanaged.cloneContext(new_allocator, new_ctx);
            return other.promoteContext(new_allocator, new_ctx);
        }

        /// Set the map to an empty state, making deinitialization a no-op, and
        /// returning a copy of the original.
        pub fn move(self: *Self) Self {
            const result = self.*;
            self.unmanaged = .{};
            return result;
        }

        /// Helper debug function to view the keys nicely
        /// when using the testing allocator
        fn printKeys(self: *Self) void {
            // not possible: assert(self.allocator == std.testing.allocator);
            const keys = self.unmanaged.header().keys[0..self.capacity()];
            const formatted_keys = @as([]Key(K), @ptrCast(keys));
            std.debug.print("{any}\n", .{formatted_keys});
        }
    };
}

/// A HashMap based on open addressing and grouped quadratic probing.
/// A lookup or modification typically incurs only 2 cache misses.
/// No order is guaranteed and any modification invalidates live iterators.
/// It achieves good performance with quite high load factors (by default,
/// grow is triggered at 80% full) and only one byte of overhead per element.
/// The struct itself is only 16 bytes for a small footprint. This comes at
/// the price of handling size with u32, which should be reasonable enough
/// for almost all uses.
/// Deletions are achieved with tombstones.
pub fn SwissHashMapUnmanaged(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime max_load_percentage: u64,
) type {
    if (max_load_percentage <= 0 or max_load_percentage >= 100)
        @compileError("max_load_percentage must be between 0 and 100.");
    return struct {
        const Self = @This();

        comptime {
            verifyContext(Context, K, K, u64, false);
        }

        // This is actually a midway pointer to the single buffer containing
        // a `Header` field, the `Metadata`s and `Entry`s.
        // At `-@sizeOf(Header)` is the Header field.
        // At `sizeOf(Metadata) * capacity + offset`, which is pointed to by
        // self.header().entries, is the array of entries.
        // This means that the hashmap only holds one live allocation, to
        // reduce memory fragmentation and struct size.
        /// Pointer to the metadata.
        metadata: ?[*]Metadata = null,

        /// Current number of elements in the hashmap.
        size: Size = 0,

        // Having a countdown to grow reduces the number of instructions to
        // execute when determining if the hashmap has enough capacity already.
        /// Number of available slots before a grow is needed to satisfy the
        /// `max_load_percentage`.
        available: Size = 0,

        // TODO: Make this equal to the group width
        // This is purely empirical and not a /very smart magic constant™/.
        /// Capacity of the first grow when bootstrapping the hashmap.
        const minimal_capacity = 16;

        // This hashmap is specially designed for sizes that fit in a u32.
        pub const Size = u32;

        // u64 hashes guarantee us that the fingerprint bits will never be used
        // to compute the index of a slot, maximizing the use of entropy.
        pub const Hash = u64;

        pub const Entry = struct {
            key_ptr: *K,
            value_ptr: *V,
        };

        pub const KV = struct {
            key: K,
            value: V,
        };

        const Header = struct {
            values: [*]V,
            keys: [*]K,
            capacity: Size,
        };

        /// Metadata for a slot. It can be in three states: empty, used or
        /// tombstone. Tombstones indicate that an entry was previously used,
        /// they are a simple way to handle removal.
        /// To this state, we add 7 bits from the slot's key hash. These are
        /// used as a fast way to disambiguate between entries without
        /// having to use the equality function. If two fingerprints are
        /// different, we know that we don't have to compare the keys at all.
        /// The 7 bits are the highest ones from a 64 bit hash. This way, not
        /// only we use the `log2(capacity)` lowest bits from the hash to determine
        /// a slot index, but we use 7 more bits to quickly resolve collisions
        /// when multiple elements with different hashes end up wanting to be in the same slot.
        /// Not using the equality function means we don't have to read into
        /// the entries array, likely avoiding a cache miss and a potentially
        /// costly function call.
        const Metadata = packed struct {
            const FingerPrint = u7;

            const free: FingerPrint = 0;
            const tombstone: FingerPrint = 1;

            fingerprint: FingerPrint = free,
            used: u1 = 0,

            const slot_free = @as(u8, @bitCast(Metadata{ .fingerprint = free }));
            const slot_tombstone = @as(u8, @bitCast(Metadata{ .fingerprint = tombstone }));

            pub fn isUsed(self: Metadata) bool {
                return self.used == 1;
            }

            pub fn isTombstone(self: Metadata) bool {
                return @as(u8, @bitCast(self)) == slot_tombstone;
            }

            pub fn isFree(self: Metadata) bool {
                return @as(u8, @bitCast(self)) == slot_free;
            }

            pub fn takeFingerprint(hash: Hash) FingerPrint {
                const hash_bits = @typeInfo(Hash).Int.bits;
                const fp_bits = @typeInfo(FingerPrint).Int.bits;
                return @as(FingerPrint, @truncate(hash >> (hash_bits - fp_bits)));
            }

            pub fn fill(self: *Metadata, fp: FingerPrint) void {
                self.used = 1;
                self.fingerprint = fp;
            }

            pub fn remove(self: *Metadata) void {
                self.used = 0;
                self.fingerprint = tombstone;
            }

            // used for debugging purposes
            pub fn format(value: @This(), comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
                if (value.isFree()) {
                    try writer.print("{s}", .{[_]u8{ ' ', ' ' }});
                } else if (value.isTombstone()) {
                    try writer.print("{s}", .{[_]u8{ 'T', 'T' }});
                } else {
                    try writer.print("{x:0>2}", .{@as(u8, @bitCast(value))});
                }
                return;
            }
        };

        comptime {
            assert(@sizeOf(Metadata) == 1);
            assert(@alignOf(Metadata) == 1);
        }

        /// A Metadata slice that is checked in parallel
        pub const Group = struct {
            // TODO: Make this selectable manually or automatically.
            const width = 16;

            comptime {
                // TODO: Add more validation here
                assert(math.isPowerOfTwo(width));
            }

            const u8s = @Vector(width, u8);
            const i8s = @Vector(width, i8);

            data: u8s,

            pub fn init(ptr: [*]Metadata) @This() {
                return .{ .data = @as([*]u8, @ptrCast(ptr))[0..width].* };
            }

            pub const BitMask = packed struct {
                const mask_type = @Type(.{ .Int = .{
                    .signedness = .unsigned,
                    .bits = width,
                } });

                mask: mask_type,

                const zero = @as(mask_type, 0);

                /// Checks if there are any matches (if any bit is set).
                inline fn hasMatch(self: @This()) bool {
                    return self.mask != @This().zero;
                }

                /// Returns the index of the lowest set bit.
                fn lowestSetBit(self: @This()) ?usize {
                    if (!self.hasMatch()) {
                        return null;
                    }
                    return @ctz(self.mask);
                }

                /// Removes the lowest set bit.
                fn removeLowestBit(self: *@This()) void {
                    self.mask &= self.mask - 1;
                }

                /// Iterator over the indices of set bits in a BitMask.
                pub const Iterator = struct {
                    mask: BitMask,

                    fn next(self: *@This()) ?usize {
                        if (self.mask.lowestSetBit()) |idx| {
                            self.mask.removeLowestBit();
                            return idx;
                        }
                        return null;
                    }
                };

                fn iterator(self: @This()) @This().Iterator {
                    return .{ .mask = self };
                }

                /// Helper function for debugging.
                fn inner(self: @This()) mask_type {
                    return @bitCast(self);
                }

                // used for debugging purposes
                pub fn format(value: @This(), comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
                    return try writer.print("{b:0>16}", .{value.mask});
                }
            };

            /// Creates a BitMask from matching bytes in the Group's Metadata slice.
            fn matchByte(self: @This(), byte: u8) BitMask {
                const cmp = @as(i8s, @splat(@bitCast(byte))) == @as(i8s, @bitCast(self.data));
                return @as(BitMask, @bitCast(cmp));
            }
            fn hasByte(self: @This(), byte: u8) bool {
                return self.matchByte(byte).hasMatch();
            }

            fn matchAvailable(self: @This()) BitMask {
                const used: u8 = 0b10000000;
                const cmp = @as(u8s, @splat(used)) > self.data;
                return @as(BitMask, @bitCast(cmp));
            }
            fn hasAvailable(self: @This()) bool {
                return self.matchAvailable().hasMatch();
            }
            fn getAvailable(self: @This()) ?usize {
                return self.matchAvailable().lowestSetBit();
            }

            fn matchFree(self: @This()) BitMask {
                return self.matchByte(Metadata.slot_free);
            }
            fn hasFree(self: @This()) bool {
                return self.matchFree().hasMatch();
            }
            fn getFree(self: @This()) ?usize {
                return self.matchFree().lowestSetBit();
            }

            fn matchFingerprint(self: @This(), fp: u7) BitMask {
                const metadata = Metadata{
                    .used = 1,
                    .fingerprint = fp,
                };
                return self.matchByte(@as(u8, @bitCast(metadata)));
            }
            fn hasFingerprint(self: @This(), fp: u7) bool {
                return self.matchFingerprint(fp).hasMatch();
            }

            fn matchTombstone(self: @This()) BitMask {
                return self.matchByte(Metadata.slot_tombstone);
            }
            fn hasTombstone(self: @This()) bool {
                return self.matchTombstone().hasMatch();
            }
            fn getTombstone(self: @This()) ?usize {
                return self.matchByte(Metadata.slot_tombstone).lowestSetBit();
            }

            // used for debugging purposes
            pub fn format(value: @This(), comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
                const metadata: [width]u8 = value.data;
                return try writer.print("{any}", .{@as([width]Metadata, @bitCast(metadata))});
            }
        };

        pub const Iterator = struct {
            hm: *const Self,
            index: Size = 0,

            pub fn next(it: *Iterator) ?Entry {
                assert(it.index <= it.hm.capacity());
                if (it.hm.size == 0) return null;

                const cap = it.hm.capacity();
                const end = it.hm.metadata.? + cap;
                var metadata = it.hm.metadata.? + it.index;

                while (metadata != end) : ({
                    metadata += 1;
                    it.index += 1;
                }) {
                    if (metadata[0].isUsed()) {
                        const key = &it.hm.keys()[it.index];
                        const value = &it.hm.values()[it.index];
                        it.index += 1;
                        return Entry{ .key_ptr = key, .value_ptr = value };
                    }
                }

                return null;
            }
        };

        pub const KeyIterator = FieldIterator(K);
        pub const ValueIterator = FieldIterator(V);

        fn FieldIterator(comptime T: type) type {
            return struct {
                len: usize,
                metadata: [*]const Metadata,
                items: [*]T,

                pub fn next(self: *@This()) ?*T {
                    while (self.len > 0) {
                        self.len -= 1;
                        const used = self.metadata[0].isUsed();
                        const item = &self.items[0];
                        self.metadata += 1;
                        self.items += 1;
                        if (used) {
                            return item;
                        }
                    }
                    return null;
                }
            };
        }

        pub const GetOrPutResult = struct {
            key_ptr: *K,
            value_ptr: *V,
            found_existing: bool,
        };

        pub const Managed = SwissHashMap(K, V, Context, max_load_percentage);

        pub fn promote(self: Self, allocator: Allocator) Managed {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call promoteContext instead.");
            return promoteContext(self, allocator, undefined);
        }

        pub fn promoteContext(self: Self, allocator: Allocator, ctx: Context) Managed {
            return .{
                .unmanaged = self,
                .allocator = allocator,
                .ctx = ctx,
            };
        }

        fn isUnderMaxLoadPercentage(size: Size, cap: Size) bool {
            return size * 100 < max_load_percentage * cap;
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.deallocate(allocator);
            self.* = undefined;
        }

        fn capacityForSize(size: Size) Size {
            var new_cap: u32 = @intCast((@as(u64, size) * 100) / max_load_percentage + 1);
            new_cap = math.ceilPowerOfTwo(u32, new_cap) catch unreachable;
            return new_cap;
        }

        pub fn ensureTotalCapacity(self: *Self, allocator: Allocator, new_size: Size) Allocator.Error!void {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call ensureTotalCapacityContext instead.");
            return ensureTotalCapacityContext(self, allocator, new_size, undefined);
        }
        pub fn ensureTotalCapacityContext(self: *Self, allocator: Allocator, new_size: Size, ctx: Context) Allocator.Error!void {
            if (new_size > self.size)
                try self.growIfNeeded(allocator, new_size - self.size, ctx);
        }

        pub fn ensureUnusedCapacity(self: *Self, allocator: Allocator, additional_size: Size) Allocator.Error!void {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call ensureUnusedCapacityContext instead.");
            return ensureUnusedCapacityContext(self, allocator, additional_size, undefined);
        }
        pub fn ensureUnusedCapacityContext(self: *Self, allocator: Allocator, additional_size: Size, ctx: Context) Allocator.Error!void {
            return ensureTotalCapacityContext(self, allocator, self.count() + additional_size, ctx);
        }

        pub fn clearRetainingCapacity(self: *Self) void {
            if (self.metadata) |_| {
                self.initMetadatas();
                self.size = 0;
                self.available = @as(u32, @truncate((self.capacity() * max_load_percentage) / 100));
            }
        }

        pub fn clearAndFree(self: *Self, allocator: Allocator) void {
            self.deallocate(allocator);
            self.size = 0;
            self.available = 0;
        }

        pub fn count(self: *const Self) Size {
            return self.size;
        }

        fn header(self: *const Self) *Header {
            return @ptrCast(@as([*]Header, @ptrCast(@alignCast(self.metadata.?))) - 1);
        }

        fn keys(self: *const Self) [*]K {
            return self.header().keys;
        }

        fn values(self: *const Self) [*]V {
            return self.header().values;
        }

        pub fn capacity(self: *const Self) Size {
            if (self.metadata == null) return 0;

            return self.header().capacity;
        }

        pub fn iterator(self: *const Self) Iterator {
            return .{ .hm = self };
        }

        pub fn keyIterator(self: *const Self) KeyIterator {
            if (self.metadata) |metadata| {
                return .{
                    .len = self.capacity(),
                    .metadata = metadata,
                    .items = self.keys(),
                };
            } else {
                return .{
                    .len = 0,
                    .metadata = undefined,
                    .items = undefined,
                };
            }
        }

        pub fn valueIterator(self: *const Self) ValueIterator {
            if (self.metadata) |metadata| {
                return .{
                    .len = self.capacity(),
                    .metadata = metadata,
                    .items = self.values(),
                };
            } else {
                return .{
                    .len = 0,
                    .metadata = undefined,
                    .items = undefined,
                };
            }
        }

        /// Insert an entry in the map. Assumes it is not already present.
        pub fn putNoClobber(self: *Self, allocator: Allocator, key: K, value: V) Allocator.Error!void {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call putNoClobberContext instead.");
            return self.putNoClobberContext(allocator, key, value, undefined);
        }
        pub fn putNoClobberContext(self: *Self, allocator: Allocator, key: K, value: V, ctx: Context) Allocator.Error!void {
            assert(!self.containsContext(key, ctx));
            try self.growIfNeeded(allocator, 1, ctx);

            self.putAssumeCapacityNoClobberContext(key, value, ctx);
        }

        /// Asserts there is enough capacity to store the new key-value pair.
        /// Clobbers any existing data. To detect if a put would clobber
        /// existing data, see `getOrPutAssumeCapacity`.
        pub fn putAssumeCapacity(self: *Self, key: K, value: V) void {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call putAssumeCapacityContext instead.");
            return self.putAssumeCapacityContext(key, value, undefined);
        }
        pub fn putAssumeCapacityContext(self: *Self, key: K, value: V, ctx: Context) void {
            const gop = self.getOrPutAssumeCapacityContext(key, ctx);
            gop.value_ptr.* = value;
        }

        /// Insert an entry in the map. Assumes it is not already present,
        /// and that no allocation is needed.
        pub fn putAssumeCapacityNoClobber(self: *Self, key: K, value: V) void {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call putAssumeCapacityNoClobberContext instead.");
            return self.putAssumeCapacityNoClobberContext(key, value, undefined);
        }
        pub fn putAssumeCapacityNoClobberContext(self: *Self, key: K, value: V, ctx: Context) void {
            assert(!self.containsContext(key, ctx));

            const hash = ctx.hash(key);
            const mask = self.capacity() - 1;
            var idx = @as(usize, @truncate(hash & mask));

            var stride: u32 = 1;
            var group = Group.init(self.metadata.? + idx);
            while (!group.hasAvailable()) {
                idx = (idx + stride) & mask;
                stride += 1;
                group = Group.init(self.metadata.? + idx);
            }

            assert(self.available > 0);
            self.available -= 1;

            const fingerprint = Metadata.takeFingerprint(hash);
            const elem_idx = idx + group.getAvailable().?;
            assert(elem_idx < self.capacity());
            (self.metadata.? + elem_idx)[0].fill(fingerprint);
            self.keys()[elem_idx] = key;
            self.values()[elem_idx] = value;

            self.size += 1;

            // std.debug.print("put key {} at index {}\n", .{ key, elem_idx });
        }

        /// Inserts a new `Entry` into the hash map, returning the previous one, if any.
        pub fn fetchPut(self: *Self, allocator: Allocator, key: K, value: V) Allocator.Error!?KV {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call fetchPutContext instead.");
            return self.fetchPutContext(allocator, key, value, undefined);
        }
        pub fn fetchPutContext(self: *Self, allocator: Allocator, key: K, value: V, ctx: Context) Allocator.Error!?KV {
            const gop = try self.getOrPutContext(allocator, key, ctx);
            var result: ?KV = null;
            if (gop.found_existing) {
                result = KV{
                    .key = gop.key_ptr.*,
                    .value = gop.value_ptr.*,
                };
            }
            gop.value_ptr.* = value;
            return result;
        }

        /// Inserts a new `Entry` into the hash map, returning the previous one, if any.
        /// If insertion happens, asserts there is enough capacity without allocating.
        pub fn fetchPutAssumeCapacity(self: *Self, key: K, value: V) ?KV {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call fetchPutAssumeCapacityContext instead.");
            return self.fetchPutAssumeCapacityContext(key, value, undefined);
        }
        pub fn fetchPutAssumeCapacityContext(self: *Self, key: K, value: V, ctx: Context) ?KV {
            const gop = self.getOrPutAssumeCapacityContext(key, ctx);
            var result: ?KV = null;
            if (gop.found_existing) {
                result = KV{
                    .key = gop.key_ptr.*,
                    .value = gop.value_ptr.*,
                };
            }
            gop.value_ptr.* = value;
            return result;
        }

        /// If there is an `Entry` with a matching key, it is deleted from
        /// the hash map, and then returned from this function.
        pub fn fetchRemove(self: *Self, key: K) ?KV {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call fetchRemoveContext instead.");
            return self.fetchRemoveContext(key, undefined);
        }
        pub fn fetchRemoveContext(self: *Self, key: K, ctx: Context) ?KV {
            return self.fetchRemoveAdapted(key, ctx);
        }
        pub fn fetchRemoveAdapted(self: *Self, key: anytype, ctx: anytype) ?KV {
            if (self.getIndex(key, ctx)) |idx| {
                const old_key = &self.keys()[idx];
                const old_val = &self.values()[idx];
                const result = KV{
                    .key = old_key.*,
                    .value = old_val.*,
                };
                self.metadata.?[idx].remove();
                old_key.* = undefined;
                old_val.* = undefined;
                self.size -= 1;
                self.available += 1;
                return result;
            }

            return null;
        }

        /// Find the index containing the data for the given key.
        /// Whether this function returns null is almost always
        /// branched on after this function returns, and this function
        /// returns null/not null from separate code paths.  We
        /// want the optimizer to remove that branch and instead directly
        /// fuse the basic blocks after the branch to the basic blocks
        /// from this function.  To encourage that, this function is
        /// marked as inline.
        inline fn getIndex(self: Self, key: anytype, ctx: anytype) ?usize {
            comptime verifyContext(@TypeOf(ctx), @TypeOf(key), K, Hash, false);

            if (self.size == 0) {
                return null;
            }

            // If you get a compile error on this line, it means that your generic hash
            // function is invalid for these parameters.
            const hash = ctx.hash(key);
            // verifyContext can't verify the return type of generic hash functions,
            // so we need to double-check it here.
            if (@TypeOf(hash) != Hash) {
                @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic hash function that returns the wrong type! " ++ @typeName(Hash) ++ " was expected, but found " ++ @typeName(@TypeOf(hash)));
            }

            const mask = self.capacity() - 1; // bucket mask
            const fingerprint = Metadata.takeFingerprint(hash);
            // std.debug.print("getIndex: fp = {x:0>2} => b = {x:0>2}\n", .{ fingerprint, @as(u8, fingerprint) ^ 0b10000000 });

            // Don't loop indefinitely when there are no empty slots.
            var limit = self.capacity();
            var idx = @as(usize, @truncate(hash & mask));
            var stride: u32 = 1;

            while (limit != 0) : ({
                // WARNING: this continue block is dependant on the `continue` expression below
                // if it is removed, then said expression also needs to be modified.
                limit -= 1;
                idx = (idx + stride) & mask;
                stride += 1; // quadratic probing
            }) {
                const group = Group.init(self.metadata.? + idx);
                // std.debug.print("idx = {}: G{any}\n", .{ idx, group });
                const bitmask = group.matchFingerprint(fingerprint);
                if (!bitmask.hasMatch()) {
                    // std.debug.print("failed match, continuing\n", .{});
                    continue;
                }
                var iter = bitmask.iterator();
                while (iter.next()) |inner_idx| {
                    // The real index of the bucket
                    const elem_idx = idx + inner_idx;
                    const test_key = &self.keys()[elem_idx];
                    // If you get a compile error on this line, it means that your generic eql
                    // function is invalid for these parameters.
                    const eql = ctx.eql(key, test_key.*);
                    // verifyContext can't verify the return type of generic eql functions,
                    // so we need to double-check it here.
                    if (@TypeOf(eql) != bool) {
                        @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic eql function that returns the wrong type! bool was expected, but found " ++ @typeName(@TypeOf(eql)));
                    }
                    if (eql) {
                        return elem_idx;
                    }
                }

                if (group.hasFree()) break;
            }
            return null;
        }

        /// Helper function for debugging.
        inline fn getIndexLinear(self: Self, key: anytype, ctx: anytype) ?usize {
            const hash = ctx.hash(key);

            const mask = self.capacity() - 1; // bucket mask
            const fingerprint = Metadata.takeFingerprint(hash);

            // Don't loop indefinitely when there are no empty slots.

            var i: u32 = 0;
            while (i <= mask) : (i += 1) {
                var metadata = self.metadata.?[i];

                if (!metadata.isUsed() or metadata.fingerprint != fingerprint) {
                    continue;
                }
                const test_key = &self.keys()[i];
                const eql = ctx.eql(key, test_key.*);
                if (eql) {
                    return i;
                }
            }

            return null;
        }

        pub fn getEntry(self: Self, key: K) ?Entry {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getEntryContext instead.");
            return self.getEntryContext(key, undefined);
        }
        pub fn getEntryContext(self: Self, key: K, ctx: Context) ?Entry {
            return self.getEntryAdapted(key, ctx);
        }
        pub fn getEntryAdapted(self: Self, key: anytype, ctx: anytype) ?Entry {
            if (self.getIndex(key, ctx)) |idx| {
                return Entry{
                    .key_ptr = &self.keys()[idx],
                    .value_ptr = &self.values()[idx],
                };
            }
            return null;
        }

        /// Insert an entry if the associated key is not already present, otherwise update preexisting value.
        pub fn put(self: *Self, allocator: Allocator, key: K, value: V) Allocator.Error!void {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call putContext instead.");
            return self.putContext(allocator, key, value, undefined);
        }
        pub fn putContext(self: *Self, allocator: Allocator, key: K, value: V, ctx: Context) Allocator.Error!void {
            const result = try self.getOrPutContext(allocator, key, ctx);
            result.value_ptr.* = value;
        }

        /// Get an optional pointer to the actual key associated with adapted key, if present.
        pub fn getKeyPtr(self: Self, key: K) ?*K {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getKeyPtrContext instead.");
            return self.getKeyPtrContext(key, undefined);
        }
        pub fn getKeyPtrContext(self: Self, key: K, ctx: Context) ?*K {
            return self.getKeyPtrAdapted(key, ctx);
        }
        pub fn getKeyPtrAdapted(self: Self, key: anytype, ctx: anytype) ?*K {
            if (self.getIndex(key, ctx)) |idx| {
                return &self.keys()[idx];
            }
            return null;
        }

        /// Get a copy of the actual key associated with adapted key, if present.
        pub fn getKey(self: Self, key: K) ?K {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getKeyContext instead.");
            return self.getKeyContext(key, undefined);
        }
        pub fn getKeyContext(self: Self, key: K, ctx: Context) ?K {
            return self.getKeyAdapted(key, ctx);
        }
        pub fn getKeyAdapted(self: Self, key: anytype, ctx: anytype) ?K {
            if (self.getIndex(key, ctx)) |idx| {
                return self.keys()[idx];
            }
            return null;
        }

        /// Get an optional pointer to the value associated with key, if present.
        pub fn getPtr(self: Self, key: K) ?*V {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getPtrContext instead.");
            return self.getPtrContext(key, undefined);
        }
        pub fn getPtrContext(self: Self, key: K, ctx: Context) ?*V {
            return self.getPtrAdapted(key, ctx);
        }
        pub fn getPtrAdapted(self: Self, key: anytype, ctx: anytype) ?*V {
            if (self.getIndex(key, ctx)) |idx| {
                return &self.values()[idx];
            }
            return null;
        }

        /// Get a copy of the value associated with key, if present.
        pub fn get(self: Self, key: K) ?V {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getContext instead.");
            return self.getContext(key, undefined);
        }
        pub fn getContext(self: Self, key: K, ctx: Context) ?V {
            return self.getAdapted(key, ctx);
        }
        pub fn getAdapted(self: Self, key: anytype, ctx: anytype) ?V {
            if (self.getIndex(key, ctx)) |idx| {
                return self.values()[idx];
            }
            return null;
        }

        pub fn getOrPut(self: *Self, allocator: Allocator, key: K) Allocator.Error!GetOrPutResult {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getOrPutContext instead.");
            return self.getOrPutContext(allocator, key, undefined);
        }
        pub fn getOrPutContext(self: *Self, allocator: Allocator, key: K, ctx: Context) Allocator.Error!GetOrPutResult {
            const gop = try self.getOrPutContextAdapted(allocator, key, ctx, ctx);
            if (!gop.found_existing) {
                gop.key_ptr.* = key;
            }
            return gop;
        }
        pub fn getOrPutAdapted(self: *Self, allocator: Allocator, key: anytype, key_ctx: anytype) Allocator.Error!GetOrPutResult {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getOrPutContextAdapted instead.");
            return self.getOrPutContextAdapted(allocator, key, key_ctx, undefined);
        }
        pub fn getOrPutContextAdapted(self: *Self, allocator: Allocator, key: anytype, key_ctx: anytype, ctx: Context) Allocator.Error!GetOrPutResult {
            self.growIfNeeded(allocator, 1, ctx) catch |err| {
                // If allocation fails, try to do the lookup anyway.
                // If we find an existing item, we can return it.
                // Otherwise return the error, we could not add another.
                const index = self.getIndex(key, key_ctx) orelse return err;
                return GetOrPutResult{
                    .key_ptr = &self.keys()[index],
                    .value_ptr = &self.values()[index],
                    .found_existing = true,
                };
            };
            return self.getOrPutAssumeCapacityAdapted(key, key_ctx);
        }

        pub fn getOrPutAssumeCapacity(self: *Self, key: K) GetOrPutResult {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getOrPutAssumeCapacityContext instead.");
            return self.getOrPutAssumeCapacityContext(key, undefined);
        }
        pub fn getOrPutAssumeCapacityContext(self: *Self, key: K, ctx: Context) GetOrPutResult {
            const result = self.getOrPutAssumeCapacityAdapted(key, ctx);
            if (!result.found_existing) {
                result.key_ptr.* = key;
            }
            return result;
        }
        pub fn getOrPutAssumeCapacityAdapted(self: *Self, key: anytype, ctx: anytype) GetOrPutResult {
            comptime verifyContext(@TypeOf(ctx), @TypeOf(key), K, Hash, false);

            // If you get a compile error on this line, it means that your generic hash
            // function is invalid for these parameters.
            const hash = ctx.hash(key);
            // verifyContext can't verify the return type of generic hash functions,
            // so we need to double-check it here.
            if (@TypeOf(hash) != Hash) {
                @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic hash function that returns the wrong type! " ++ @typeName(Hash) ++ " was expected, but found " ++ @typeName(@TypeOf(hash)));
            }

            const mask = self.capacity() - 1;
            const fingerprint = Metadata.takeFingerprint(hash);

            var limit = self.capacity();
            var idx = @as(usize, @truncate(hash & mask));
            var stride: u32 = 1;

            var first_tombstone_idx: usize = self.capacity(); // invalid index
            var group = Group.init(self.metadata.? + idx);
            while (limit != 0) : ({
                limit -= 1;
                idx = (idx + stride) & mask;
                stride += 1; // quadratic probing
                group = Group.init(self.metadata.? + idx);
            }) {
                const bit_mask = group.matchFingerprint(fingerprint);
                if (bit_mask.hasMatch()) {
                    var iter = bit_mask.iterator();
                    while (iter.next()) |inner_idx| {
                        const elem_idx = idx + inner_idx;
                        const test_key = &self.keys()[elem_idx];
                        // If you get a compile error on this line, it means that your generic eql
                        // function is invalid for these parameters.
                        const eql = ctx.eql(key, test_key.*);
                        // verifyContext can't verify the return type of generic eql functions,
                        // so we need to double-check it here.
                        if (@TypeOf(eql) != bool) {
                            @compileError("Context " ++ @typeName(@TypeOf(ctx)) ++ " has a generic eql function that returns the wrong type! bool was expected, but found " ++ @typeName(@TypeOf(eql)));
                        }
                        if (eql) {
                            return GetOrPutResult{
                                .key_ptr = test_key,
                                .value_ptr = &self.values()[elem_idx],
                                .found_existing = true,
                            };
                        }
                    }
                } else if (first_tombstone_idx == self.capacity() and group.getTombstone() != null) {
                    first_tombstone_idx = idx + group.getTombstone().?;
                }

                if (group.hasFree()) break;
            }

            const elem_idx = if (group.getFree()) |free_idx| @min(idx + free_idx, first_tombstone_idx) else first_tombstone_idx;
            // We're using a slot previously free or a tombstone.
            self.available -= 1;
            self.metadata.?[elem_idx].fill(fingerprint);
            const new_key = &self.keys()[elem_idx];
            const new_value = &self.values()[elem_idx];
            new_key.* = undefined;
            new_value.* = undefined;
            self.size += 1;

            return GetOrPutResult{
                .key_ptr = new_key,
                .value_ptr = new_value,
                .found_existing = false,
            };
        }

        pub fn getOrPutValue(self: *Self, allocator: Allocator, key: K, value: V) Allocator.Error!Entry {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call getOrPutValueContext instead.");
            return self.getOrPutValueContext(allocator, key, value, undefined);
        }
        pub fn getOrPutValueContext(self: *Self, allocator: Allocator, key: K, value: V, ctx: Context) Allocator.Error!Entry {
            const res = try self.getOrPutAdapted(allocator, key, ctx);
            if (!res.found_existing) {
                res.key_ptr.* = key;
                res.value_ptr.* = value;
            }
            return Entry{ .key_ptr = res.key_ptr, .value_ptr = res.value_ptr };
        }

        /// Return true if there is a value associated with key in the map.
        pub fn contains(self: *const Self, key: K) bool {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call containsContext instead.");
            return self.containsContext(key, undefined);
        }
        pub fn containsContext(self: *const Self, key: K, ctx: Context) bool {
            return self.containsAdapted(key, ctx);
        }
        pub fn containsAdapted(self: *const Self, key: anytype, ctx: anytype) bool {
            return self.getIndex(key, ctx) != null;
        }

        fn removeByIndex(self: *Self, idx: usize) void {
            self.metadata.?[idx].remove();
            self.keys()[idx] = undefined;
            self.values()[idx] = undefined;
            self.size -= 1;
            self.available += 1;
        }

        /// If there is an `Entry` with a matching key, it is deleted from
        /// the hash map, and this function returns true.  Otherwise this
        /// function returns false.
        pub fn remove(self: *Self, key: K) bool {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call removeContext instead.");
            return self.removeContext(key, undefined);
        }
        pub fn removeContext(self: *Self, key: K, ctx: Context) bool {
            return self.removeAdapted(key, ctx);
        }
        pub fn removeAdapted(self: *Self, key: anytype, ctx: anytype) bool {
            if (self.getIndex(key, ctx)) |idx| {
                self.removeByIndex(idx);
                return true;
            }

            return false;
        }

        /// Delete the entry with key pointed to by key_ptr from the hash map.
        /// key_ptr is assumed to be a valid pointer to a key that is present
        /// in the hash map.
        pub fn removeByPtr(self: *Self, key_ptr: *K) void {
            // TODO: replace with pointer subtraction once supported by zig
            // if @sizeOf(K) == 0 then there is at most one item in the hash
            // map, which is assumed to exist as key_ptr must be valid.  This
            // item must be at index 0.
            const idx = if (@sizeOf(K) > 0)
                (@intFromPtr(key_ptr) - @intFromPtr(self.keys())) / @sizeOf(K)
            else
                0;

            self.removeByIndex(idx);
        }

        fn initMetadatas(self: *Self) void {
            const end = @sizeOf(Metadata) * self.capacity();
            // set the normal metadata to empty
            @memset(@as([*]u8, @ptrCast(self.metadata.?))[0..end], 0);
            // set the special end metadata group (can't be tombstone due to putAssumeCapacityNoClobber)
            @memset(@as([*]u8, @ptrCast(self.metadata.?))[end .. end + Group.width], 0b10101010);
        }

        // This counts the number of occupied slots (not counting tombstones), which is
        // what has to stay under the max_load_percentage of capacity.
        fn load(self: *const Self) Size {
            const max_load = (self.capacity() * max_load_percentage) / 100;
            assert(max_load >= self.available);
            return @as(Size, @truncate(max_load - self.available));
        }

        fn growIfNeeded(self: *Self, allocator: Allocator, new_count: Size, ctx: Context) Allocator.Error!void {
            if (new_count > self.available) {
                try self.grow(allocator, capacityForSize(self.load() + new_count), ctx);
            }
        }

        pub fn clone(self: Self, allocator: Allocator) Allocator.Error!Self {
            if (@sizeOf(Context) != 0)
                @compileError("Cannot infer context " ++ @typeName(Context) ++ ", call cloneContext instead.");
            return self.cloneContext(allocator, @as(Context, undefined));
        }
        pub fn cloneContext(self: Self, allocator: Allocator, new_ctx: anytype) Allocator.Error!SwissHashMapUnmanaged(K, V, @TypeOf(new_ctx), max_load_percentage) {
            var other = SwissHashMapUnmanaged(K, V, @TypeOf(new_ctx), max_load_percentage){};
            if (self.size == 0)
                return other;

            const new_cap = capacityForSize(self.size);
            try other.allocate(allocator, new_cap);
            other.initMetadatas();
            other.available = @truncate((new_cap * max_load_percentage) / 100);

            var i: Size = 0;
            var metadata = self.metadata.?;
            const keys_ptr = self.keys();
            const values_ptr = self.values();
            while (i < self.capacity()) : (i += 1) {
                if (metadata[i].isUsed()) {
                    other.putAssumeCapacityNoClobberContext(keys_ptr[i], values_ptr[i], new_ctx);
                    if (other.size == self.size)
                        break;
                }
            }

            return other;
        }

        /// Set the map to an empty state, making deinitialization a no-op, and
        /// returning a copy of the original.
        pub fn move(self: *Self) Self {
            const result = self.*;
            self.* = .{};
            return result;
        }

        fn grow(self: *Self, allocator: Allocator, new_capacity: Size, ctx: Context) Allocator.Error!void {
            @setCold(true);
            const new_cap = @max(new_capacity, minimal_capacity);
            assert(new_cap > self.capacity());
            assert(std.math.isPowerOfTwo(new_cap));

            var map = Self{};
            defer map.deinit(allocator);
            try map.allocate(allocator, new_cap);
            map.initMetadatas();
            map.available = @truncate((new_cap * max_load_percentage) / 100);

            if (self.size != 0) {
                const old_capacity = self.capacity();
                var i: Size = 0;
                var metadata = self.metadata.?;
                const keys_ptr = self.keys();
                const values_ptr = self.values();
                while (i < old_capacity) : (i += 1) {
                    if (metadata[i].isUsed()) {
                        map.putAssumeCapacityNoClobberContext(keys_ptr[i], values_ptr[i], ctx);
                        if (map.size == self.size)
                            break;
                    }
                }
            }

            self.size = 0;
            std.mem.swap(Self, self, &map);
        }

        fn allocate(self: *Self, allocator: Allocator, new_capacity: Size) Allocator.Error!void {
            const header_align = @alignOf(Header);
            const key_align = if (@sizeOf(K) == 0) 1 else @alignOf(K);
            const val_align = if (@sizeOf(V) == 0) 1 else @alignOf(V);
            const max_align = comptime @max(header_align, key_align, val_align);

            const new_cap: usize = new_capacity;
            // Include the final tombstone group
            const meta_size = @sizeOf(Header) + (new_cap + Group.width) * @sizeOf(Metadata);
            comptime assert(@alignOf(Metadata) == 1);

            const keys_start = std.mem.alignForward(usize, meta_size, key_align);
            const keys_end = keys_start + new_cap * @sizeOf(K);

            const vals_start = std.mem.alignForward(usize, keys_end, val_align);
            const vals_end = vals_start + new_cap * @sizeOf(V);

            const total_size = std.mem.alignForward(usize, vals_end, max_align);

            const slice = try allocator.alignedAlloc(u8, max_align, total_size);
            const ptr = @intFromPtr(slice.ptr);

            const metadata = ptr + @sizeOf(Header);

            const hdr = @as(*Header, @ptrFromInt(ptr));
            if (@sizeOf([*]V) != 0) {
                hdr.values = @as([*]V, @ptrFromInt(ptr + vals_start));
            }
            if (@sizeOf([*]K) != 0) {
                hdr.keys = @as([*]K, @ptrFromInt(ptr + keys_start));
            }
            hdr.capacity = new_capacity;
            self.metadata = @as([*]Metadata, @ptrFromInt(metadata));
        }

        fn deallocate(self: *Self, allocator: Allocator) void {
            if (self.metadata == null) return;

            const header_align = @alignOf(Header);
            const key_align = if (@sizeOf(K) == 0) 1 else @alignOf(K);
            const val_align = if (@sizeOf(V) == 0) 1 else @alignOf(V);
            const max_align = comptime @max(header_align, key_align, val_align);

            const cap: usize = self.capacity();
            // Account for the tombstone group here: vvvvvvvvvvvvv
            const meta_size = @sizeOf(Header) + (cap + Group.width) * @sizeOf(Metadata);
            comptime assert(@alignOf(Metadata) == 1);

            const keys_start = std.mem.alignForward(usize, meta_size, key_align);
            const keys_end = keys_start + cap * @sizeOf(K);

            const vals_start = std.mem.alignForward(usize, keys_end, val_align);
            const vals_end = vals_start + cap * @sizeOf(V);

            const total_size = std.mem.alignForward(usize, vals_end, max_align);

            const slice = @as([*]align(max_align) u8, @ptrFromInt(@intFromPtr(self.header())))[0..total_size];
            allocator.free(slice);

            self.metadata = null;
            self.available = 0;
        }

        /// This function is used in the debugger pretty formatters in tools/ to fetch the
        /// header type to facilitate fancy debug printing for this type.
        fn dbHelper(self: *Self, hdr: *Header, entry: *Entry) void {
            _ = self;
            _ = hdr;
            _ = entry;
        }

        comptime {
            if (!builtin.strip_debug_info) {
                _ = &dbHelper;
            }
        }
    };
}

fn Key(key: type) type {
    return struct {
        val: key,

        pub fn format(value: @This(), comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
            if (value.val == 2863311530) {
                try writer.print(" ", .{});
            } else {
                try writer.print("{}", .{value.val});
            }
        }
    };
}

const testing = std.testing;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

// zwizz tests
// test "zwizz test" {
//     const Metadata = AutoSwissHashMap(u32, u32).Unmanaged.Metadata;
//     const metadata: Metadata = .{
//         .used = 1,
//         .fingerprint = 0,
//     };
//     std.debug.print("\n{}\n", .{@as(u8, @bitCast(metadata))});
// }
test "zwizz basic" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    const count = 5;
    var i: u32 = 1;
    while (i < count) : (i += 2) {
        try map.put(i, i);
    }

    var iter = map.valueIterator();
    try expectEqual(iter.next().?.*, @as(u32, 1));
    try expectEqual(iter.next().?.*, @as(u32, 3));
    try expectEqual(iter.next(), null);
}
test "zwizz groups" {
    const Unmanaged = AutoSwissHashMap(u32, u32).Unmanaged;
    const Metadata = Unmanaged.Metadata;
    const Group = Unmanaged.Group;

    const b: u8 = 0x0a;
    const arr: [16]u8 = [_]u8{ 0, 0, 0, 0, 0, 0, b, 0, 0, 0, 1, 1, 1, 1, 1, 1 };
    const metadata = @as([*]Metadata, @ptrCast(@constCast(&arr[0])));
    var group = Group.init(metadata);

    const match = group.matchByte(b);
    try expectEqual(match.hasMatch(), true);
    var iter = match.iterator();
    try expectEqual(iter.next().?, 6);
    try expectEqual(iter.next(), null);
}
// test "zwizz core" {
//     const Map = AutoSwissHashMap(u32, u32);
//     const U = Map.Unmanaged;
//     var map = Map.init(std.testing.allocator);
//     defer map.deinit();
//
//     try map.put(1, 1);
//     std.debug.print("{any}\n", .{@as([*]u8, @ptrCast(map.unmanaged.metadata.?))[0 .. map.capacity() + Map.Unmanaged.Group.width]});
//
//     const hash = AutoContext(u32).hash(undefined, 1);
//     std.debug.print("hash: {any}\n", .{hash});
//
//     std.debug.print("fingerprint: {any}\n", .{U.Metadata.takeFingerprint(hash)});
//
//     const mask = map.capacity() - 1;
//     const idx = @as(usize, @truncate(hash & mask));
//
//     std.debug.print("index: {any}\n", .{idx});
//     std.debug.print("metadata: {any}\n", .{map.unmanaged.metadata.?[idx]});
// }

// original tests
test "basic usage" {
    // std.debug.print("\n", .{});
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    const count = 5;
    var i: u32 = 0;
    var total: u32 = 0;
    while (i < count) : (i += 1) {
        try map.put(i, i);
        total += i;
    }

    var sum: u32 = 0;
    var it = map.iterator();
    while (it.next()) |kv| {
        sum += kv.key_ptr.*;
    }
    try expectEqual(total, sum);

    i = 0;
    sum = 0;
    while (i < count) : (i += 1) {
        sum += map.get(i).?;
    }
    try expectEqual(total, sum);
}

test "ensureTotalCapacity" {
    var map = AutoSwissHashMap(i32, i32).init(std.testing.allocator);
    defer map.deinit();

    try map.ensureTotalCapacity(20);
    const initial_capacity = map.capacity();
    try testing.expect(initial_capacity >= 20);
    var i: i32 = 0;
    while (i < 20) : (i += 1) {
        try testing.expect(map.fetchPutAssumeCapacity(i, i + 10) == null);
    }
    // shouldn't resize from putAssumeCapacity
    try testing.expect(initial_capacity == map.capacity());
}

test "ensureUnusedCapacity with tombstones" {
    var map = AutoSwissHashMap(i32, i32).init(std.testing.allocator);
    defer map.deinit();

    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        try map.ensureUnusedCapacity(1);
        map.putAssumeCapacity(i, i);
        _ = map.remove(i);
    }
}

test "clearRetainingCapacity" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    map.clearRetainingCapacity();

    try map.put(1, 1);
    try expectEqual(map.get(1).?, 1);
    try expectEqual(map.count(), 1);

    map.clearRetainingCapacity();
    map.putAssumeCapacity(1, 1);
    try expectEqual(map.get(1).?, 1);
    try expectEqual(map.count(), 1);

    const cap = map.capacity();
    try expect(cap > 0);

    map.clearRetainingCapacity();
    map.clearRetainingCapacity();
    try expectEqual(map.count(), 0);
    try expectEqual(map.capacity(), cap);
    try expect(!map.contains(1));
}

// when it grows its still putting the elements in linear order and not quadratic (?)
test "grow" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    const growTo = 12456;

    var i: u32 = 0;
    while (i < growTo) : (i += 1) {
        try map.put(i, i);
    }
    try expectEqual(growTo, map.count());

    // std.debug.print("\n  <puts done>\n", .{});

    i = 0;
    var it = map.iterator();
    while (it.next()) |kv| {
        try expectEqual(kv.key_ptr.*, kv.value_ptr.*);
        i += 1;
    }
    try expectEqual(i, growTo);

    // std.debug.print("\n  <iter done>\n", .{});

    // const idx = map.unmanaged.getIndexLinear(861, map.ctx);
    // std.debug.print("861@{any}\n", .{idx});

    i = 860;
    while (i < growTo) : (i += 1) {
        // std.debug.print("getting {}\n", .{i});
        try expectEqual(map.get(i).?, i);
    }
}

test "clone" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var a = try map.clone();
    defer a.deinit();

    try expectEqual(a.count(), 0);

    try a.put(1, 1);
    try a.put(2, 2);
    try a.put(3, 3);

    var b = try a.clone();
    defer b.deinit();

    try expectEqual(b.count(), 3);
    try expectEqual(b.get(1).?, 1);
    try expectEqual(b.get(2).?, 2);
    try expectEqual(b.get(3).?, 3);

    var original = AutoSwissHashMap(i32, i32).init(std.testing.allocator);
    defer original.deinit();

    var i: u8 = 0;
    while (i < 10) : (i += 1) {
        try original.putNoClobber(i, i * 10);
    }

    var copy = try original.clone();
    defer copy.deinit();

    i = 0;
    while (i < 10) : (i += 1) {
        try testing.expect(copy.get(i).? == i * 10);
    }
}

test "ensureTotalCapacity with existing elements" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.put(0, 0);
    try expectEqual(map.count(), 1);
    try expectEqual(map.capacity(), @TypeOf(map).Unmanaged.minimal_capacity);

    try map.ensureTotalCapacity(65);
    try expectEqual(map.count(), 1);
    try expectEqual(map.capacity(), 128);
}

test "ensureTotalCapacity satisfies max load factor" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.ensureTotalCapacity(127);
    try expectEqual(map.capacity(), 256);
}

test "remove" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        try map.put(i, i);
    }

    i = 0;
    while (i < 16) : (i += 1) {
        if (i % 3 == 0) {
            _ = map.remove(i);
        }
    }
    try expectEqual(map.count(), 10);
    var it = map.iterator();
    while (it.next()) |kv| {
        try expectEqual(kv.key_ptr.*, kv.value_ptr.*);
        try expect(kv.key_ptr.* % 3 != 0);
    }

    i = 0;
    while (i < 16) : (i += 1) {
        if (i % 3 == 0) {
            try expect(!map.contains(i));
        } else {
            try expectEqual(map.get(i).?, i);
        }
    }
}

test "reverse removes" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        try map.putNoClobber(i, i);
    }

    i = 16;
    while (i > 0) : (i -= 1) {
        _ = map.remove(i - 1);
        try expect(!map.contains(i - 1));
        var j: u32 = 0;
        while (j < i - 1) : (j += 1) {
            try expectEqual(map.get(j).?, j);
        }
    }

    try expectEqual(map.count(), 0);
}

test "multiple removes on same metadata" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        try map.put(i, i);
    }

    _ = map.remove(7);
    _ = map.remove(15);
    _ = map.remove(14);
    _ = map.remove(13);
    try expect(!map.contains(7));
    try expect(!map.contains(15));
    try expect(!map.contains(14));
    try expect(!map.contains(13));

    i = 0;
    while (i < 13) : (i += 1) {
        if (i == 7) {
            try expect(!map.contains(i));
        } else {
            try expectEqual(map.get(i).?, i);
        }
    }

    try map.put(15, 15);
    try map.put(13, 13);
    try map.put(14, 14);
    try map.put(7, 7);
    i = 0;
    while (i < 16) : (i += 1) {
        try expectEqual(map.get(i).?, i);
    }
}

test "put and remove loop in random order" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var keys = std.ArrayList(u32).init(std.testing.allocator);
    defer keys.deinit();

    const size = 32;
    const iterations = 100;

    var i: u32 = 0;
    while (i < size) : (i += 1) {
        try keys.append(i);
    }
    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    while (i < iterations) : (i += 1) {
        random.shuffle(u32, keys.items);

        for (keys.items) |key| {
            try map.put(key, key);
        }
        try expectEqual(map.count(), size);

        for (keys.items) |key| {
            _ = map.remove(key);
        }
        try expectEqual(map.count(), 0);
    }
}

test "remove one million elements in random order" {
    const Map = AutoSwissHashMap(u32, u32);
    const n = 1000 * 1000;
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

// This test will fail as it checks the linear probing in the std implementation
// TODO: Replace it with the equivalent in the quadratic probing case
//
// test "put" {
//     var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
//     defer map.deinit();
//
//     var i: u32 = 0;
//     while (i < 16) : (i += 1) {
//         try map.put(i, i);
//     }
//
//     i = 0;
//     while (i < 16) : (i += 1) {
//         try expectEqual(map.get(i).?, i);
//     }
//
//     i = 0;
//     while (i < 16) : (i += 1) {
//         try map.put(i, i * 16 + 1);
//     }
//
//     i = 0;
//     while (i < 16) : (i += 1) {
//         try expectEqual(map.get(i).?, i * 16 + 1);
//     }
// }

test "putAssumeCapacity" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.ensureTotalCapacity(20);
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        map.putAssumeCapacityNoClobber(i, i);
    }

    i = 0;
    var sum = i;
    while (i < 20) : (i += 1) {
        sum += map.getPtr(i).?.*;
    }
    try expectEqual(190, sum);

    i = 0;
    while (i < 20) : (i += 1) {
        map.putAssumeCapacity(i, 1);
    }

    i = 0;
    sum = i;
    while (i < 20) : (i += 1) {
        sum += map.get(i).?;
    }

    try expectEqual(20, sum);
}

test "repeat putAssumeCapacity/remove" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    try map.ensureTotalCapacity(20);
    const limit = map.unmanaged.available;
    // std.debug.print("\nlimit = {}\n", .{limit});
    // std.debug.print("\ninitial put\n", .{});

    var i: u32 = 0;
    while (i < limit) : (i += 1) {
        map.putAssumeCapacityNoClobber(i, i);
    }

    // map.printKeys();

    // Repeatedly delete/insert an entry without resizing the map.
    // Put to different keys so entries don't land in the just-freed slot.
    i = 0;
    while (i < 10 * limit) : (i += 1) {
        // std.debug.print("remove key = {}, left with s={}, a={}\n", .{ i, map.unmanaged.size, map.unmanaged.available + 1 });
        try testing.expect(map.remove(i));
        // std.debug.print("{any} \n", .{map.unmanaged.metadata.?[0..map.unmanaged.size]});
        // map.printKeys();
        if (i % 2 == 0) {
            // std.debug.print("-> pACNC with {}\n", .{limit + i});
            map.putAssumeCapacityNoClobber(limit + i, i);
            // map.printKeys();
        } else {
            // std.debug.print("-> pAC with {}\n", .{limit + i});
            map.putAssumeCapacity(limit + i, i);
            // map.printKeys();
        }
    }

    // std.debug.print("pre get\n", .{});

    i = 9 * limit;
    while (i < 10 * limit) : (i += 1) {
        try expectEqual(map.get(limit + i), i);
    }
    try expectEqual(map.unmanaged.available, 0);
    try expectEqual(map.unmanaged.count(), limit);
}

test "getOrPut" {
    var map = AutoSwissHashMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        try map.put(i * 2, 2);
    }

    i = 0;
    while (i < 20) : (i += 1) {
        _ = try map.getOrPutValue(i, 1);
    }

    i = 0;
    var sum = i;
    while (i < 20) : (i += 1) {
        sum += map.get(i).?;
    }

    try expectEqual(sum, 30);
}

test "basic hash map usage" {
    var map = AutoSwissHashMap(i32, i32).init(std.testing.allocator);
    defer map.deinit();

    try testing.expect((try map.fetchPut(1, 11)) == null);
    try testing.expect((try map.fetchPut(2, 22)) == null);
    try testing.expect((try map.fetchPut(3, 33)) == null);
    try testing.expect((try map.fetchPut(4, 44)) == null);

    try map.putNoClobber(5, 55);
    try testing.expect((try map.fetchPut(5, 66)).?.value == 55);
    try testing.expect((try map.fetchPut(5, 55)).?.value == 66);

    const gop1 = try map.getOrPut(5);
    try testing.expect(gop1.found_existing == true);
    try testing.expect(gop1.value_ptr.* == 55);
    gop1.value_ptr.* = 77;
    try testing.expect(map.getEntry(5).?.value_ptr.* == 77);

    const gop2 = try map.getOrPut(99);
    try testing.expect(gop2.found_existing == false);
    gop2.value_ptr.* = 42;
    try testing.expect(map.getEntry(99).?.value_ptr.* == 42);

    const gop3 = try map.getOrPutValue(5, 5);
    try testing.expect(gop3.value_ptr.* == 77);

    const gop4 = try map.getOrPutValue(100, 41);
    try testing.expect(gop4.value_ptr.* == 41);

    try testing.expect(map.contains(2));
    try testing.expect(map.getEntry(2).?.value_ptr.* == 22);
    try testing.expect(map.get(2).? == 22);

    const rmv1 = map.fetchRemove(2);
    try testing.expect(rmv1.?.key == 2);
    try testing.expect(rmv1.?.value == 22);
    try testing.expect(map.fetchRemove(2) == null);
    try testing.expect(map.remove(2) == false);
    try testing.expect(map.getEntry(2) == null);
    try testing.expect(map.get(2) == null);

    try testing.expect(map.remove(3) == true);
}

test "getOrPutAdapted" {
    const AdaptedContext = struct {
        pub fn eql(self: @This(), adapted_key: []const u8, test_key: u64) bool {
            _ = self;
            return std.fmt.parseInt(u64, adapted_key, 10) catch unreachable == test_key;
        }
        pub fn hash(self: @This(), adapted_key: []const u8) u64 {
            _ = self;
            const key = std.fmt.parseInt(u64, adapted_key, 10) catch unreachable;
            return (AutoContext(u64){}).hash(key);
        }
    };
    var map = AutoSwissHashMap(u64, u64).init(testing.allocator);
    defer map.deinit();

    const keys = [_][]const u8{
        "1231",
        "4564",
        "7894",
        "1132",
        "65235",
        "95462",
        "0112305",
        "00658",
        "0",
        "2",
    };

    var real_keys: [keys.len]u64 = undefined;

    inline for (keys, 0..) |key_str, i| {
        const result = try map.getOrPutAdapted(key_str, AdaptedContext{});
        try testing.expect(!result.found_existing);
        real_keys[i] = std.fmt.parseInt(u64, key_str, 10) catch unreachable;
        result.key_ptr.* = real_keys[i];
        result.value_ptr.* = i * 2;
    }

    try testing.expectEqual(map.count(), keys.len);

    inline for (keys, 0..) |key_str, i| {
        const result = map.getOrPutAssumeCapacityAdapted(key_str, AdaptedContext{});
        try testing.expect(result.found_existing);
        try testing.expectEqual(real_keys[i], result.key_ptr.*);
        try testing.expectEqual(@as(u64, i) * 2, result.value_ptr.*);
        try testing.expectEqual(real_keys[i], map.getKeyAdapted(key_str, AdaptedContext{}).?);
    }
}

test "ensureUnusedCapacity" {
    var map = AutoSwissHashMap(u64, u64).init(testing.allocator);
    defer map.deinit();

    try map.ensureUnusedCapacity(32);
    const capacity = map.capacity();
    try map.ensureUnusedCapacity(32);

    // Repeated ensureUnusedCapacity() calls with no insertions between
    // should not change the capacity.
    try testing.expectEqual(capacity, map.capacity());
}

test "removeByPtr" {
    var map = AutoSwissHashMap(i32, u64).init(testing.allocator);
    defer map.deinit();

    var i: i32 = undefined;

    i = 0;
    while (i < 10) : (i += 1) {
        try map.put(i, 0);
    }

    try testing.expect(map.count() == 10);

    i = 0;
    while (i < 10) : (i += 1) {
        const key_ptr = map.getKeyPtr(i);
        try testing.expect(key_ptr != null);

        if (key_ptr) |ptr| {
            map.removeByPtr(ptr);
        }
    }

    try testing.expect(map.count() == 0);
}

test "removeByPtr 0 sized key" {
    var map = AutoSwissHashMap(u0, u64).init(testing.allocator);
    defer map.deinit();

    try map.put(0, 0);

    try testing.expect(map.count() == 1);

    const key_ptr = map.getKeyPtr(0);
    try testing.expect(key_ptr != null);

    if (key_ptr) |ptr| {
        map.removeByPtr(ptr);
    }

    try testing.expect(map.count() == 0);
}

test "repeat fetchRemove" {
    var map = AutoSwissHashMapUnmanaged(u64, void){};
    defer map.deinit(testing.allocator);

    try map.ensureTotalCapacity(testing.allocator, 4);

    map.putAssumeCapacity(0, {});
    map.putAssumeCapacity(1, {});
    map.putAssumeCapacity(2, {});
    map.putAssumeCapacity(3, {});

    // fetchRemove() should make slots available.
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try testing.expect(map.fetchRemove(3) != null);
        map.putAssumeCapacity(3, {});
    }

    try testing.expect(map.get(0) != null);
    try testing.expect(map.get(1) != null);
    try testing.expect(map.get(2) != null);
    try testing.expect(map.get(3) != null);
}
