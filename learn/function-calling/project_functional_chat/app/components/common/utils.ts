export function stringifyObject(obj: any, indentLevel: number = 0): string {
  const indent = ' '.repeat(indentLevel * 4); // 4 spaces per indent level
  const subIndent = ' '.repeat((indentLevel + 1) * 4);

  if (Array.isArray(obj)) {
    return '[\n' + obj.map(item => subIndent + stringifyObject(item, indentLevel + 1)).join(',\n') + '\n' + indent + ']';
  } else if (typeof obj === 'object' && obj !== null) {
    return '{\n' + Object.entries(obj).map(([key, value]) =>
      `${subIndent}${key}: ${stringifyObject(value, indentLevel + 1)}`).join(',\n') + '\n' + indent + '}';
  } else {
    return JSON.stringify(obj);
  }
}
