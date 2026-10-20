// Export Ghidra decompiler High P-Code with basic-block CFG as JSONL.
// @category ReGraph

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.data.Array;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.Pointer;
import ghidra.program.model.data.TypeDef;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
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

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

public class PcodeDump extends GhidraScript {

	@Override
	public void run() throws Exception {
		if (currentProgram == null) {
			printerr("No current program is loaded.");
			return;
		}

		String[] args = getScriptArgs();
		if (args.length < 1) {
			throw new IllegalArgumentException(
				"Usage: PcodeDump.java <output.jsonl> [function_limit] [decompile_timeout_seconds]"
			);
		}

		String outputPath = args[0];
		int functionLimit = parseIntArg(args, 1, 0);
		int timeout = parseIntArg(args, 2, 60);

		DecompInterface decompiler = new DecompInterface();
		DecompileOptions options = new DecompileOptions();
		options.grabFromProgram(currentProgram);
		decompiler.setOptions(options);
		decompiler.toggleCCode(false);
		decompiler.toggleSyntaxTree(true);
		decompiler.setSimplificationStyle("decompile");
		if (!decompiler.openProgram(currentProgram)) {
			throw new IllegalStateException("Unable to initialize Ghidra decompiler: " +
				decompiler.getLastMessage());
		}

		int emitted = 0;
		int failed = 0;
		FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
		try (BufferedWriter writer = new BufferedWriter(
			new OutputStreamWriter(new FileOutputStream(outputPath), StandardCharsets.UTF_8))) {
			while (functions.hasNext()) {
				monitor.checkCancelled();
				Function function = functions.next();
				if (function == null || function.isExternal()) {
					continue;
				}
				if (functionLimit > 0 && emitted >= functionLimit) {
					break;
				}

				DecompileResults results = decompiler.decompileFunction(function, timeout, monitor);
				if (results == null || !results.decompileCompleted() || results.getHighFunction() == null) {
					writeFailure(writer, function,
						results == null ? "no decompile result" : results.getErrorMessage());
					writer.write("\n");
					failed++;
					emitted++;
					continue;
				}

				writeFunction(writer, results.getHighFunction());
				writer.write("\n");
				emitted++;
			}
		}
		finally {
			decompiler.dispose();
		}

		printf("Exported %d function records to %s (%d decompile failures)%n",
			emitted, outputPath, failed);
	}

	private int parseIntArg(String[] args, int index, int defaultValue) {
		if (index >= args.length || args[index] == null || args[index].isEmpty()) {
			return defaultValue;
		}
		return Integer.parseInt(args[index]);
	}

	private void writeFailure(BufferedWriter writer, Function function, String error) throws Exception {
		writer.write("{");
		writer.write("\"program\":");
		writeProgram(writer);
		writer.write(",");
		writeStringField(writer, "name", function.getName(), true);
		writeStringField(writer, "signature", function.getSignature().toString(), true);
		writeStringField(writer, "entry", function.getEntryPoint().toString(), true);
		writeStringField(writer, "entry_offset", Long.toUnsignedString(function.getEntryPoint().getOffset()), true);
		writeStringField(writer, "decompile_error", error == null ? "" : error, false);
		writer.write("}");
	}

	private void writeFunction(BufferedWriter writer, HighFunction highFunction) throws Exception {
		Function function = highFunction.getFunction();
		writer.write("{");
		writer.write("\"program\":");
		writeProgram(writer);
		writer.write(",");
		writeStringField(writer, "name", function.getName(), true);
		writeStringField(writer, "signature", function.getSignature().toString(), true);
		writeStringField(writer, "entry", function.getEntryPoint().toString(), true);
		writeStringField(writer, "entry_offset", Long.toUnsignedString(function.getEntryPoint().getOffset()), true);

		FunctionPrototype prototype = highFunction.getFunctionPrototype();
		writer.write("\"return_type\":");
		writeDataType(writer, prototype == null ? null : prototype.getReturnType());
		writer.write(",");
		writer.write("\"parameters\":[");
		if (prototype != null) {
			for (int i = 0; i < prototype.getNumParams(); i++) {
				if (i > 0) {
					writer.write(",");
				}
				writeParameter(writer, highFunction, prototype.getParam(i), i);
			}
		}
		writer.write("],");

		writer.write("\"blocks\":[");
		boolean firstBlock = true;
		for (PcodeBlockBasic block : highFunction.getBasicBlocks()) {
			if (!firstBlock) {
				writer.write(",");
			}
			writeBlock(writer, highFunction, block);
			firstBlock = false;
		}
		writer.write("]");
		writer.write("}");
	}

	private void writeProgram(BufferedWriter writer) throws Exception {
		writer.write("{");
		writeStringField(writer, "name", currentProgram.getName(), true);
		writeStringField(writer, "executable_path", currentProgram.getExecutablePath(), true);
		writeStringField(writer, "language", currentProgram.getLanguage().getLanguageID().getIdAsString(), true);
		writeStringField(writer, "compiler", currentProgram.getCompilerSpec().getCompilerSpecID().getIdAsString(), true);
		writeStringField(writer, "image_base", currentProgram.getImageBase().toString(), true);
		writeStringField(writer, "image_base_offset", Long.toUnsignedString(currentProgram.getImageBase().getOffset()), true);
		writeIntField(writer, "pointer_size", currentProgram.getDefaultPointerSize(), true);
		writeBoolField(writer, "big_endian", currentProgram.getLanguage().isBigEndian(), false);
		writer.write("}");
	}

	private void writeParameter(
		BufferedWriter writer, HighFunction highFunction, HighSymbol parameter, int index)
			throws Exception {
		writer.write("{");
		writeIntField(writer, "index", index, true);
		if (parameter == null) {
			writeStringField(writer, "name", "arg" + index, true);
			writer.write("\"type\":");
			writeDataType(writer, null);
			writer.write(",\"representative\":null}");
			return;
		}
		writeStringField(writer, "name", parameter.getName(), true);
		HighVariable highVariable = parameter.getHighVariable();
		DataType dataType = highVariable == null ? null : highVariable.getDataType();
		writer.write("\"type\":");
		writeDataType(writer, dataType);
		writer.write(",\"representative\":");
		if (highVariable == null || highVariable.getRepresentative() == null) {
			writer.write("null");
		}
		else {
			writeVarnode(writer, highFunction, highVariable.getRepresentative());
		}
		writer.write("}");
	}

	private void writeBlock(BufferedWriter writer, HighFunction highFunction, PcodeBlockBasic block)
			throws Exception {
		writer.write("{");
		writeIntField(writer, "index", block.getIndex(), true);
		writeStringField(writer, "start", block.getStart() == null ? "" : block.getStart().toString(), true);
		writer.write("\"out\":[");
		for (int i = 0; i < block.getOutSize(); i++) {
			if (i > 0) {
				writer.write(",");
			}
			PcodeBlock out = block.getOut(i);
			writer.write(Integer.toString(out.getIndex()));
		}
		writer.write("],");
		writer.write("\"in\":[");
		for (int i = 0; i < block.getInSize(); i++) {
			if (i > 0) {
				writer.write(",");
			}
			PcodeBlock in = block.getIn(i);
			writer.write(Integer.toString(in.getIndex()));
		}
		writer.write("],");
		writer.write("\"true_out\":");
		writer.write(block.getOutSize() > 1 && block.getTrueOut() != null
			? Integer.toString(block.getTrueOut().getIndex())
			: "null");
		writer.write(",\"false_out\":");
		writer.write(block.getOutSize() > 1 && block.getFalseOut() != null
			? Integer.toString(block.getFalseOut().getIndex())
			: "null");
		writer.write(",\"ops\":[");
		Iterator<PcodeOp> ops = block.getIterator();
		int opIndex = 0;
		while (ops.hasNext()) {
			PcodeOp op = ops.next();
			if (op == null) {
				continue;
			}
			if (opIndex > 0) {
				writer.write(",");
			}
			writePcodeOp(writer, highFunction, op, opIndex);
			opIndex++;
		}
		writer.write("]}");
	}

	private void writePcodeOp(
		BufferedWriter writer, HighFunction highFunction, PcodeOp op, int blockOpIndex)
			throws Exception {
		writer.write("{");
		writeIntField(writer, "block_op_index", blockOpIndex, true);
		writeStringField(writer, "mnemonic", op.getMnemonic(), true);
		writeIntField(writer, "opcode", op.getOpcode(), true);
		writeStringField(writer, "text", op.toString(), true);
		SequenceNumber seq = op.getSeqnum();
		writeStringField(writer, "seq_target", seq == null ? "" : seq.getTarget().toString(), true);
		writeIntField(writer, "seq_time", seq == null ? 0 : seq.getTime(), true);
		writer.write("\"output\":");
		Varnode output = op.getOutput();
		if (output == null) {
			writer.write("null");
		}
		else {
			writeVarnode(writer, highFunction, output);
		}
		writer.write(",\"inputs\":[");
		for (int i = 0; i < op.getNumInputs(); i++) {
			if (i > 0) {
				writer.write(",");
			}
			writeVarnode(writer, highFunction, op.getInput(i));
		}
		writer.write("]}");
	}

	private void writeVarnode(BufferedWriter writer, HighFunction highFunction, Varnode varnode)
			throws Exception {
		writer.write("{");
		writeStringField(writer, "repr", varnode.toString(), true);
		writeStringField(writer, "address", varnode.getAddress().toString(), true);
		writeStringField(writer, "space", varnode.getAddress().getAddressSpace().getName(), true);
		writeStringField(writer, "offset", Long.toUnsignedString(varnode.getOffset()), true);
		writeIntField(writer, "size", varnode.getSize(), true);
		writeBoolField(writer, "is_constant", varnode.isConstant(), true);
		writeBoolField(writer, "is_address", varnode.isAddress(), true);
		writeBoolField(writer, "is_register", varnode.isRegister(), true);
		writeBoolField(writer, "is_unique", varnode.isUnique(), true);
		writeBoolField(writer, "is_addr_tied", varnode.isAddrTied(), true);
		writeBoolField(writer, "is_persistent", varnode.isPersistent(), true);
		writeBoolField(writer, "is_hash", varnode.isHash(), true);
		writeStringField(writer, "pc_address", getPcAddress(varnode), true);
		HighVariable highVariable = null;
		if (varnode instanceof VarnodeAST) {
			highVariable = ((VarnodeAST) varnode).getHigh();
		}
		if (highVariable == null) {
			writer.write("\"high\":null");
		}
		else {
			writer.write("\"high\":{");
			writeStringField(writer, "name", highVariable.getName(), true);
			writeIntField(writer, "size", highVariable.getSize(), true);
			writer.write("\"type\":");
			writeDataType(writer, highVariable.getDataType());
			writer.write("}");
		}
		writer.write("}");
	}

	private String getPcAddress(Varnode varnode) {
		if (varnode instanceof VarnodeAST) {
			VarnodeAST ast = (VarnodeAST) varnode;
			return ast.getPCAddress() == null ? "" : ast.getPCAddress().toString();
		}
		return "";
	}

	private void writeDataType(BufferedWriter writer, DataType dataType) throws Exception {
		if (dataType == null) {
			writer.write("{\"name\":\"unknown\",\"length\":0,\"kind\":\"unknown\"}");
			return;
		}
		DataType base = dataType;
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
		else if (base.getName().toLowerCase().contains("float") ||
				base.getName().toLowerCase().contains("double")) {
			kind = "float";
		}
		writer.write("{");
		writeStringField(writer, "name", base.getName(), true);
		writeStringField(writer, "display", base.getDisplayName(), true);
		writeIntField(writer, "length", base.getLength(), true);
		writeStringField(writer, "kind", kind, false);
		writer.write("}");
	}

	private void writeStringField(BufferedWriter writer, String name, String value, boolean comma)
			throws Exception {
		writer.write("\"");
		writer.write(escape(name));
		writer.write("\":\"");
		writer.write(escape(value == null ? "" : value));
		writer.write("\"");
		if (comma) {
			writer.write(",");
		}
	}

	private void writeIntField(BufferedWriter writer, String name, int value, boolean comma)
			throws Exception {
		writer.write("\"");
		writer.write(escape(name));
		writer.write("\":");
		writer.write(Integer.toString(value));
		if (comma) {
			writer.write(",");
		}
	}

	private void writeBoolField(BufferedWriter writer, String name, boolean value, boolean comma)
			throws Exception {
		writer.write("\"");
		writer.write(escape(name));
		writer.write("\":");
		writer.write(value ? "true" : "false");
		if (comma) {
			writer.write(",");
		}
	}

	private String escape(String value) {
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < value.length(); i++) {
			char ch = value.charAt(i);
			switch (ch) {
				case '\\':
					builder.append("\\\\");
					break;
				case '"':
					builder.append("\\\"");
					break;
				case '\b':
					builder.append("\\b");
					break;
				case '\f':
					builder.append("\\f");
					break;
				case '\n':
					builder.append("\\n");
					break;
				case '\r':
					builder.append("\\r");
					break;
				case '\t':
					builder.append("\\t");
					break;
				default:
					if (ch < 0x20) {
						builder.append(String.format("\\u%04x", (int) ch));
					}
					else {
						builder.append(ch);
					}
			}
		}
		return builder.toString();
	}
}
