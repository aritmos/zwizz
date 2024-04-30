alias pl := probelen

probelen file:
    awk -F ',' '{ sum += $3; count++ } END { if (count > 0) print "Average:", sum / count; else print "No data found" }' {{file}}

sim:
    rm -f zwz_uq.dat zwz_mock.dat
    zig test --test-filter "zwizz basic" src/swiss_hash_map.zig &> zwz_uq.dat
    zig test --test-filter "zwizz basic" src/mock_swiss_hash_map.zig &> zwz_mock.dat
    diff -y zwz_uq.dat zwz_mock.dat

count file:
     awk '/^-> / {sum += $2 } END { print sum }' {{file}}
    
counts:
    just count zwz_uq.dat
    just count zwz_mock.dat

alias tf := transfer

windir := '/mnt/c/Users/Sebastian/Desktop/zwizz'
transfer:
    cp src/main.zig {{windir}}/src/main.zig
    cp src/swiss_hash_map.zig {{windir}}/src/swiss_hash_map.zig

alias b := build
build name:
    zig build-exe -O ReleaseFast --name {{name}} src/main.zig
