const joinFileNames = (files) => files.map((name) => `"${name}"`).join(' ');

export default {
  'ts/**/*.ts?(x)': () => 'yarn ts:check:all',
  'ts/**/*.{js,jsx,ts,tsx}': (files) => `yarn ts:lint:fix ${joinFileNames(files)}`,
};
