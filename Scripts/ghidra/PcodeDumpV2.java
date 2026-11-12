// Export Ghidra P-Code (decompiler High P-Code, or raw per-instruction P-Code) plus the program facts the
// lifter needs to symbolize memory (blocks + bytes, data references, functions, symbols) as JSONL.
// Line 1 is a "program" record; every further line is one function record.
// Usage: PcodeDumpV2.java <out.jsonl> <fva_file|-> <decompile_timeout_s> <high|raw>
//   fva_file: one function entry address (hex, image base as loaded) per line; "-" = all non-external functions.
// @category ReGraph

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressIterator;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.data.Array;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.Pointer;
import ghidra.program.model.data.TypeDef;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.pcode.FunctionPrototype;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighSymbol;
import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.PcodeBlock;
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.SequenceNumber;
import ghidra.program.model.pcode.Varnode;
import ghidra.program.model.pcode.VarnodeAST;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.Symbol;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class PcodeDumpV2 extends GhidraScript {

	private static final long MAX_BLOCK_BYTES = 256L * 1024 * 1024;

	@Override
	public void run() throws Exception {
		String[] args = getScriptArgs();
		if (args.length < 4) {
			throw new IllegalArgumentException("Usage: PcodeDumpV2.java <out.jsonl> <fva_file|-> <timeout_s> <high|raw>");
		}
		String outputPath = args[0];
		Set<Long> wanted = null;
		if (!args[1].equals("-")) {
			wanted = new HashSet<>();
			for (String line : Files.readAllLines(Paths.get(args[1]))) {
				line = line.trim();
				if (line.isEmpty()) {
					continue;
				}
				wanted.add(Long.parseUnsignedLong(line.startsWith("0x") ? line.substring(2) : line, 16));
			}
		}
		int timeout = Integer.parseInt(args[2]);
		String mode = args[3];

		DecompInterface decompiler = null;
		if (mode.equals("high")) {
			decompiler = new DecompInterface();
			DecompileOptions options = new DecompileOptions();
			options.grabFromProgram(currentProgram);
			decompiler.setOptions(options);
			decompiler.toggleCCode(false);
			decompiler.toggleSyntaxTree(true);
			decompiler.setSimplificationStyle("decompile");
			if (!decompiler.openProgram(currentProgram)) {
				throw new IllegalStateException("decompiler init failed: " + decompiler.getLastMessage());
			}
		}

		int emitted = 0, failed = 0;
		try (BufferedWriter w = new BufferedWriter(
			new OutputStreamWriter(new FileOutputStream(outputPath), StandardCharsets.UTF_8))) {
			writeProgramRecord(w);
			w.write("\n");
			FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
			Set<Long> seen = new HashSet<>();
			while (functions.hasNext()) {
				monitor.checkCancelled();
				Function f = functions.next();
				if (f == null || f.isExternal()) {
					continue;
				}
				long entry = f.getEntryPoint().getOffset();
				if (wanted != null && !wanted.contains(entry)) {
					continue;
				}
				seen.add(entry);
				if (mode.equals("high")) {
					DecompileResults res = decompiler.decompileFunction(f, timeout, monitor);
					if (res == null || !res.decompileCompleted() || res.getHighFunction() == null) {
						writeFailure(w, f, res == null ? "no decompile result" : res.getErrorMessage());
						failed++;
					}
					else {
						writeHighFunction(w, res.getHighFunction());
					}
				}
				else {
					writeRawFunction(w, f);
				}
				w.write("\n");
				emitted++;
			}
			if (wanted != null) {
				for (Long fva : wanted) {
					if (!seen.contains(fva)) {
						w.write("{\"kind\":\"missing\",\"entry_offset\":\"" + Long.toUnsignedString(fva) + "\"}\n");
					}
				}
			}
		}
		finally {
			if (decompiler != null) {
				decompiler.dispose();
			}
		}
		printf("PcodeDumpV2: %d functions (%d decompile failures) -> %s%n", emitted, failed, outputPath);
	}

	// ---------------------------------------------------------------- program record

	private void writeProgramRecord(BufferedWriter w) throws Exception {
		w.write("{\"kind\":\"program\",");
		str(w, "name", currentProgram.getName(), true);
		str(w, "language", currentProgram.getLanguage().getLanguageID().getIdAsString(), true);
		str(w, "compiler", currentProgram.getCompilerSpec().getCompilerSpecID().getIdAsString(), true);
		str(w, "processor", currentProgram.getLanguage().getProcessor().toString(), true);
		str(w, "image_base_offset", Long.toUnsignedString(currentProgram.getImageBase().getOffset()), true);
		num(w, "pointer_size", currentProgram.getDefaultPointerSize(), true);
		bool(w, "big_endian", currentProgram.getLanguage().isBigEndian(), true);
		Varnode sp = currentProgram.getCompilerSpec().getStackPointer() == null ? null :
			new Varnode(currentProgram.getCompilerSpec().getStackPointer().getAddress(),
				currentProgram.getCompilerSpec().getStackPointer().getMinimumByteSize());
		w.write("\"stack_pointer\":");
		if (sp == null) {
			w.write("null,");
		}
		else {
			w.write("{");
			str(w, "name", currentProgram.getCompilerSpec().getStackPointer().getName(), true);
			str(w, "offset", Long.toUnsignedString(sp.getOffset()), true);
			num(w, "size", sp.getSize(), false);
			w.write("},");
		}

		// memory blocks
		w.write("\"blocks\":[");
		boolean first = true;
		for (MemoryBlock b : currentProgram.getMemory().getBlocks()) {
			if (!first) {
				w.write(",");
			}
			first = false;
			w.write("{");
			str(w, "name", b.getName(), true);
			str(w, "space", b.getStart().getAddressSpace().getName(), true);
			str(w, "start", Long.toUnsignedString(b.getStart().getOffset()), true);
			str(w, "size", Long.toUnsignedString(b.getSize()), true);
			bool(w, "initialized", b.isInitialized(), true);
			bool(w, "read", b.isRead(), true);
			bool(w, "write", b.isWrite(), true);
			bool(w, "execute", b.isExecute(), true);
			bool(w, "external", b.isExternalBlock(), true);
			boolean dumpBytes = b.isInitialized() && b.getStart().isMemoryAddress() && b.getSize() <= MAX_BLOCK_BYTES;
			// initialized but not exported (too large / non-memory space): the consumer must reject, never zero-fill
			bool(w, "bytes_unavailable", b.isInitialized() && !dumpBytes, true);
			if (dumpBytes) {
				byte[] data = new byte[(int) b.getSize()];
				b.getBytes(b.getStart(), data);
				str(w, "bytes_b64", Base64.getEncoder().encodeToString(data), false);
			}
			else {
				str(w, "bytes_b64", "", false);
			}
			w.write("}");
		}
		w.write("],");

		// references originating in data (non-execute) memory: pointer slots to symbolize
		w.write("\"data_refs\":[");
		first = true;
		AddressSet dataSet = new AddressSet();
		for (MemoryBlock b : currentProgram.getMemory().getBlocks()) {
			if (b.isInitialized() && !b.isExecute() && b.getStart().isMemoryAddress()) {
				dataSet.add(b.getStart(), b.getEnd());
			}
		}
		AddressIterator srcs = currentProgram.getReferenceManager().getReferenceSourceIterator(dataSet, true);
		while (srcs.hasNext()) {
			Address from = srcs.next();
			for (Reference r : currentProgram.getReferenceManager().getReferencesFrom(from)) {
				Address to = r.getToAddress();
				if (!to.isMemoryAddress() && !to.isExternalAddress()) {
					continue;
				}
				if (!first) {
					w.write(",");
				}
				first = false;
				w.write("[\"" + Long.toUnsignedString(from.getOffset()) + "\",\"" +
					Long.toUnsignedString(to.getOffset()) + "\",\"" + to.getAddressSpace().getName() + "\"]");
			}
		}
		w.write("],");

		// functions (all, incl. thunks and externals), with every symbol name at the entry
		w.write("\"functions\":[");
		first = true;
		FunctionIterator all = currentProgram.getFunctionManager().getFunctions(true);
		while (all.hasNext()) {
			Function f = all.next();
			if (!first) {
				w.write(",");
			}
			first = false;
			writeFunctionInfo(w, f);
		}
		FunctionIterator ext = currentProgram.getFunctionManager().getExternalFunctions();
		while (ext.hasNext()) {
			Function f = ext.next();
			if (!first) {
				w.write(",");
			}
			first = false;
			writeFunctionInfo(w, f);
		}
		w.write("],");

		// data symbols (labels) outside code
		w.write("\"data_symbols\":[");
		first = true;
		for (Symbol s : currentProgram.getSymbolTable().getAllSymbols(false)) {
			Address a = s.getAddress();
			if (a == null || !(a.isMemoryAddress() || a.isExternalAddress())) {
				continue;
			}
			MemoryBlock b = currentProgram.getMemory().getBlock(a);
			if (b != null && b.isExecute()) {
				continue;
			}
			if (currentProgram.getFunctionManager().getFunctionAt(a) != null) {
				continue;
			}
			if (!first) {
				w.write(",");
			}
			first = false;
			w.write("[\"" + Long.toUnsignedString(a.getOffset()) + "\",\"" + escape(s.getName(true)) + "\",\"" +
				a.getAddressSpace().getName() + "\"]");
		}
		w.write("]}");
	}

	private void writeFunctionInfo(BufferedWriter w, Function f) throws Exception {
		w.write("{");
		str(w, "entry_offset", Long.toUnsignedString(f.getEntryPoint().getOffset()), true);
		str(w, "space", f.getEntryPoint().getAddressSpace().getName(), true);
		str(w, "name", f.getName(), true);
		bool(w, "external", f.isExternal(), true);
		bool(w, "thunk", f.isThunk(), true);
		bool(w, "noreturn", f.hasNoReturn(), true);
		Function t = f.isThunk() ? f.getThunkedFunction(true) : null;
		str(w, "thunk_target", t == null ? "" : Long.toUnsignedString(t.getEntryPoint().getOffset()), true);
		str(w, "thunk_target_space", t == null ? "" : t.getEntryPoint().getAddressSpace().getName(), true);
		str(w, "thunk_target_name", t == null ? "" : t.getName(), true);
		w.write("\"symbols\":[");
		boolean first = true;
		for (Symbol s : currentProgram.getSymbolTable().getSymbols(f.getEntryPoint())) {
			if (!first) {
				w.write(",");
			}
			first = false;
			w.write("\"" + escape(s.getName()) + "\"");
		}
		w.write("]}");
	}

	// ---------------------------------------------------------------- functions

	private void writeFailure(BufferedWriter w, Function f, String error) throws Exception {
		w.write("{\"kind\":\"function\",");
		str(w, "name", f.getName(), true);
		str(w, "entry_offset", Long.toUnsignedString(f.getEntryPoint().getOffset()), true);
		str(w, "decompile_error", error == null ? "" : error, false);
		w.write("}");
	}

	private void writeHighFunction(BufferedWriter w, HighFunction hf) throws Exception {
		Function f = hf.getFunction();
		w.write("{\"kind\":\"function\",\"mode\":\"high\",");
		str(w, "name", f.getName(), true);
		str(w, "entry_offset", Long.toUnsignedString(f.getEntryPoint().getOffset()), true);
		writeGp(w, f);
		FunctionPrototype proto = hf.getFunctionPrototype();
		w.write("\"return_type\":");
		writeDataType(w, proto == null ? null : proto.getReturnType());
		w.write(",\"parameters\":[");
		if (proto != null) {
			for (int i = 0; i < proto.getNumParams(); i++) {
				if (i > 0) {
					w.write(",");
				}
				HighSymbol p = proto.getParam(i);
				w.write("{");
				num(w, "index", i, true);
				str(w, "name", p == null ? "arg" + i : p.getName(), true);
				HighVariable hv = p == null ? null : p.getHighVariable();
				w.write("\"type\":");
				writeDataType(w, hv == null ? (p == null ? null : p.getDataType()) : hv.getDataType());
				w.write(",");
				num(w, "size", p == null ? 0 : p.getSize(), false);
				w.write(",\"representative\":");
				if (hv == null || hv.getRepresentative() == null) {
					w.write("null");
				}
				else {
					writeVarnode(w, hv.getRepresentative());
				}
				w.write(",\"instances\":[");
				if (hv != null) {
					boolean fi = true;
					for (Varnode inst : hv.getInstances()) {
						if (!inst.isInput()) {
							continue;
						}
						if (!fi) {
							w.write(",");
						}
						fi = false;
						writeVarnode(w, inst);
					}
				}
				w.write("]}");
			}
		}
		w.write("],\"blocks\":[");
		boolean first = true;
		for (PcodeBlockBasic b : hf.getBasicBlocks()) {
			if (!first) {
				w.write(",");
			}
			first = false;
			w.write("{");
			num(w, "index", b.getIndex(), true);
			str(w, "start", b.getStart() == null ? "" : Long.toUnsignedString(b.getStart().getOffset()), true);
			w.write("\"out\":[");
			for (int i = 0; i < b.getOutSize(); i++) {
				if (i > 0) {
					w.write(",");
				}
				w.write(Integer.toString(b.getOut(i).getIndex()));
			}
			w.write("],\"in\":[");
			for (int i = 0; i < b.getInSize(); i++) {
				if (i > 0) {
					w.write(",");
				}
				w.write(Integer.toString(b.getIn(i).getIndex()));
			}
			w.write("],\"true_out\":");
			w.write(b.getOutSize() > 1 && b.getTrueOut() != null ? Integer.toString(b.getTrueOut().getIndex()) : "null");
			w.write(",\"false_out\":");
			w.write(b.getOutSize() > 1 && b.getFalseOut() != null ? Integer.toString(b.getFalseOut().getIndex()) : "null");
			w.write(",\"ops\":[");
			Iterator<PcodeOp> it = b.getIterator();
			boolean fo = true;
			while (it.hasNext()) {
				PcodeOp op = it.next();
				if (op == null) {
					continue;
				}
				if (!fo) {
					w.write(",");
				}
				fo = false;
				writeOp(w, op);
			}
			w.write("]}");
		}
		w.write("]}");
	}

	private void writeRawFunction(BufferedWriter w, Function f) throws Exception {
		w.write("{\"kind\":\"function\",\"mode\":\"raw\",");
		str(w, "name", f.getName(), true);
		str(w, "entry_offset", Long.toUnsignedString(f.getEntryPoint().getOffset()), true);
		writeGp(w, f);
		w.write("\"instructions\":[");
		InstructionIterator it = currentProgram.getListing().getInstructions(f.getBody(), true);
		boolean first = true;
		while (it.hasNext()) {
			Instruction ins = it.next();
			if (!first) {
				w.write(",");
			}
			first = false;
			w.write("{");
			str(w, "address", Long.toUnsignedString(ins.getAddress().getOffset()), true);
			num(w, "length", ins.getLength(), true);
			str(w, "mnemonic", ins.getMnemonicString(), true);
			str(w, "text", ins.toString(), true);
			str(w, "flow", ins.getFlowType().toString(), true);
			bool(w, "in_delay_slot", ins.isInDelaySlot(), true);
			num(w, "delay_depth", ins.getDelaySlotDepth(), true);
			str(w, "fallthrough", ins.getFallThrough() == null ? "" : Long.toUnsignedString(ins.getFallThrough().getOffset()), true);
			w.write("\"flows\":[");
			Address[] flows = ins.getFlows();
			for (int i = 0; i < flows.length; i++) {
				if (i > 0) {
					w.write(",");
				}
				w.write("\"" + Long.toUnsignedString(flows[i].getOffset()) + "\"");
			}
			w.write("],\"refs\":[");
			boolean fr = true;
			for (Reference r : ins.getReferencesFrom()) {
				if (!r.getToAddress().isMemoryAddress() && !r.getToAddress().isExternalAddress()) {
					continue;
				}
				if (!fr) {
					w.write(",");
				}
				fr = false;
				w.write("[\"" + Long.toUnsignedString(r.getToAddress().getOffset()) + "\",\"" +
					r.getReferenceType().toString() + "\"]");
			}
			w.write("],\"pcode\":[");
			PcodeOp[] ops = ins.getPcode();
			for (int i = 0; i < ops.length; i++) {
				if (i > 0) {
					w.write(",");
				}
				writeOp(w, ops[i]);
			}
			w.write("]}");
		}
		w.write("]}");
	}

	private void writeGp(BufferedWriter w, Function f) throws Exception {
		Register gp = currentProgram.getRegister("gp");
		BigInteger v = null;
		if (gp != null) {
			v = currentProgram.getProgramContext().getValue(gp, f.getEntryPoint(), false);
		}
		str(w, "gp_value", v == null ? "" : v.toString(), true);
	}

	private void writeOp(BufferedWriter w, PcodeOp op) throws Exception {
		w.write("{");
		str(w, "mnemonic", op.getMnemonic(), true);
		num(w, "opcode", op.getOpcode(), true);
		str(w, "text", op.toString(), true);
		SequenceNumber seq = op.getSeqnum();
		str(w, "seq_target", seq == null ? "" : Long.toUnsignedString(seq.getTarget().getOffset()), true);
		num(w, "seq_time", seq == null ? 0 : seq.getTime(), true);
		if (op.getOpcode() == PcodeOp.CALLOTHER && op.getNumInputs() > 0 && op.getInput(0).isConstant()) {
			String name = currentProgram.getLanguage().getUserDefinedOpName((int) op.getInput(0).getOffset());
			str(w, "userop", name == null ? "" : name, true);
		}
		w.write("\"output\":");
		if (op.getOutput() == null) {
			w.write("null");
		}
		else {
			writeVarnode(w, op.getOutput());
		}
		w.write(",\"inputs\":[");
		for (int i = 0; i < op.getNumInputs(); i++) {
			if (i > 0) {
				w.write(",");
			}
			writeVarnode(w, op.getInput(i));
		}
		w.write("]}");
	}

	private void writeVarnode(BufferedWriter w, Varnode v) throws Exception {
		w.write("{");
		str(w, "repr", v.toString(), true);
		str(w, "space", v.getAddress().getAddressSpace().getName(), true);
		str(w, "offset", Long.toUnsignedString(v.getOffset()), true);
		num(w, "size", v.getSize(), true);
		bool(w, "is_constant", v.isConstant(), true);
		bool(w, "is_address", v.isAddress(), true);
		bool(w, "is_register", v.isRegister(), true);
		bool(w, "is_unique", v.isUnique(), true);
		bool(w, "is_addr_tied", v.isAddrTied(), true);
		bool(w, "is_input", v.isInput(), true);
		Register reg = v.isRegister() ? currentProgram.getRegister(v.getAddress(), v.getSize()) : null;
		if (reg == null && v.isRegister()) {
			reg = currentProgram.getRegister(v.getAddress());
		}
		str(w, "reg", reg == null ? "" : reg.getName(), true);
		if (v instanceof VarnodeAST) {
			str(w, "uid", Integer.toString(((VarnodeAST) v).getUniqueId()), true);
			HighVariable hv = ((VarnodeAST) v).getHigh();
			w.write("\"high_type\":");
			writeDataType(w, hv == null ? null : hv.getDataType());
		}
		else {
			str(w, "uid", "", true);
			w.write("\"high_type\":null");
		}
		w.write("}");
	}

	private void writeDataType(BufferedWriter w, DataType dt) throws Exception {
		if (dt == null) {
			w.write("{\"name\":\"unknown\",\"length\":0,\"kind\":\"unknown\"}");
			return;
		}
		DataType base = dt;
		while (base instanceof TypeDef) {
			base = ((TypeDef) base).getBaseDataType();
		}
		String kind = "scalar";
		if (base instanceof Pointer) {
			kind = "pointer";
		}
		else if (base instanceof Array) {
			kind = "array";
		}
		else if (base.getName().equalsIgnoreCase("void")) {
			kind = "void";
		}
		else if (base.getName().toLowerCase().contains("float") || base.getName().toLowerCase().contains("double")) {
			kind = "float";
		}
		w.write("{");
		str(w, "name", base.getName(), true);
		num(w, "length", base.getLength(), true);
		str(w, "kind", kind, false);
		w.write("}");
	}

	// ---------------------------------------------------------------- json helpers

	private void str(BufferedWriter w, String k, String v, boolean comma) throws Exception {
		w.write("\"" + escape(k) + "\":\"" + escape(v == null ? "" : v) + "\"");
		if (comma) {
			w.write(",");
		}
	}

	private void num(BufferedWriter w, String k, long v, boolean comma) throws Exception {
		w.write("\"" + escape(k) + "\":" + v);
		if (comma) {
			w.write(",");
		}
	}

	private void bool(BufferedWriter w, String k, boolean v, boolean comma) throws Exception {
		w.write("\"" + escape(k) + "\":" + (v ? "true" : "false"));
		if (comma) {
			w.write(",");
		}
	}

	private String escape(String value) {
		StringBuilder b = new StringBuilder();
		for (int i = 0; i < value.length(); i++) {
			char c = value.charAt(i);
			switch (c) {
				case '\\': b.append("\\\\"); break;
				case '"': b.append("\\\""); break;
				case '\n': b.append("\\n"); break;
				case '\r': b.append("\\r"); break;
				case '\t': b.append("\\t"); break;
				default:
					if (c < 0x20) {
						b.append(String.format("\\u%04x", (int) c));
					}
					else {
						b.append(c);
					}
			}
		}
		return b.toString();
	}
}
