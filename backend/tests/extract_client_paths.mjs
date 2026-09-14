/**
 * Prints, as one JSON object on stdout, what the TypeScript parser sees of the
 * request paths written in one source file.
 *
 * Usage: `node extract_client_paths.mjs <file.ts>`. Read by
 * `backend/tests/test_client_route_targets.py`, which is its only caller.
 *
 * The rule is the reconstructed text of a literal node, and nothing else. A
 * `StringLiteral` and a `NoSubstitutionTemplateLiteral` reconstruct to
 * `node.text`; a `TemplateExpression` reconstructs to its head text followed,
 * for each span, by one U+0000 standing for the interpolated expression and
 * then that span's literal text. A node with no text is not a candidate. This
 * is why a path written inside a block comment or a line comment is invisible
 * here: a comment is trivia, never a node, so no pattern has to be taught to
 * skip one. ADR 062 records the decision and the three rejected alternatives.
 *
 * Output shape:
 *
 * - `literals`: `{ line, text }` for every reconstructed text beginning with
 *   `/` and containing no whitespace. These are the candidate request paths.
 * - `assembled`: `{ line, text, shape }` for a path built rather than written
 *   whole — a template whose head is empty and one of whose span literals
 *   begins with `/`, or a `+` concatenation with a path-shaped operand. The
 *   reconstruction rule cannot see the path such a shape produces, so the
 *   caller fails on the finding instead of passing over it.
 * - `fetchCalls`: `{ line, headIsEmpty, spanExpressionKinds, spanIdentifiers,
 *   spanLiteralTexts }` for every call whose callee is the identifier `fetch`.
 *   Shape data rather than path data; the caller pins it so that a request
 *   composed some other way cannot appear without a test noticing.
 *
 * Exit codes: `0` with JSON on stdout; `2` with the parse diagnostics on
 * stderr when the file does not parse — `ts.createSourceFile` does not throw on
 * a broken file, it returns a tree of error nodes that yields no literals at
 * all, which would read as a clean client rather than as a broken gate; `1` on
 * a usage or I/O error.
 */

import { readFileSync } from "node:fs";
import ts from "typescript";

const file = process.argv[2];
if (!file) {
  process.stderr.write("usage: node extract_client_paths.mjs <file.ts>\n");
  process.exit(1);
}

const source = readFileSync(file, "utf8");
const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);

const diagnostics = sourceFile.parseDiagnostics ?? [];
if (diagnostics.length > 0) {
  const rendered = diagnostics.map((diagnostic) => {
    const { line } = sourceFile.getLineAndCharacterOfPosition(diagnostic.start ?? 0);
    return `${file}:${line + 1}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, " ")}`;
  });
  process.stderr.write(`${rendered.join("\n")}\n`);
  process.exit(2);
}

const INTERPOLATION = String.fromCharCode(0);

function reconstruct(node) {
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
    return node.text;
  }
  if (ts.isTemplateExpression(node)) {
    const spans = node.templateSpans.map((span) => INTERPOLATION + span.literal.text);
    return node.head.text + spans.join("");
  }
  return null;
}

function lineOf(node) {
  return sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1;
}

function isPathShaped(text) {
  return text !== null && text.startsWith("/") && !/\s/.test(text);
}

const literals = [];
const assembled = [];
const fetchCalls = [];

function recordFetchCall(node) {
  const argument = node.arguments[0];
  const isTemplate = argument !== undefined && ts.isTemplateExpression(argument);
  const spans = isTemplate ? argument.templateSpans : [];
  fetchCalls.push({
    line: lineOf(node),
    headIsEmpty: isTemplate && argument.head.text === "",
    spanExpressionKinds: spans.map((span) => ts.SyntaxKind[span.expression.kind]),
    spanIdentifiers: spans.map((span) =>
      ts.isIdentifier(span.expression) ? span.expression.text : null,
    ),
    spanLiteralTexts: spans.map((span) => span.literal.text),
  });
}

function recordAssembly(node) {
  if (ts.isTemplateExpression(node) && node.head.text === "") {
    const leading = node.templateSpans.find((span) => span.literal.text.startsWith("/"));
    if (leading !== undefined) {
      assembled.push({
        line: lineOf(node),
        text: leading.literal.text,
        shape: "template with a leading interpolation",
      });
    }
  }
  if (!ts.isBinaryExpression(node) || node.operatorToken.kind !== ts.SyntaxKind.PlusToken) {
    return;
  }
  for (const operand of [node.left, node.right]) {
    const operandText = reconstruct(operand);
    if (operandText !== null && operandText.startsWith("/")) {
      assembled.push({ line: lineOf(node), text: operandText, shape: "concatenation with +" });
    }
  }
}

function visit(node) {
  const text = reconstruct(node);
  if (isPathShaped(text)) {
    literals.push({ line: lineOf(node), text });
  }
  recordAssembly(node);
  const callee = ts.isCallExpression(node) ? node.expression : null;
  if (callee !== null && ts.isIdentifier(callee) && callee.text === "fetch") {
    recordFetchCall(node);
  }
  ts.forEachChild(node, visit);
}

visit(sourceFile);

process.stdout.write(`${JSON.stringify({ literals, assembled, fetchCalls })}\n`);
