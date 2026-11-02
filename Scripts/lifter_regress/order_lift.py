"""order_lift.py LIFTER_DIR BINARY OUT.ll MODE : run_lift with the function order normal / reverse / shuffle:SEED."""
import sys, random
sys.path.insert(0, sys.argv[1])
import ida2llvm as L
binp, out, mode = sys.argv[2], sys.argv[3], sys.argv[4]
orig = L.BIN2LLVMController.insertAllFunctions


def patched(self):
    eas = list(self._function_eas)
    if mode == "reverse":
        eas.reverse()
    elif mode.startswith("list:"):
        eas = [int(x, 16) for x in open(mode[5:]).read().split()]
    elif mode.startswith("shuffle"):
        random.Random(int(mode.split(":")[1])).shuffle(eas)
    self._function_eas = eas
    return orig(self)


L.BIN2LLVMController.insertAllFunctions = patched
r = L.run_lift(binp, out)
print("status", r["status"], r["counts"], r.get("error", ""))
