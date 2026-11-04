# Output contract

A generated page is exactly two files, returned in two fenced blocks tagged with their paths:

```jsx path=src/App.jsx
export default function App() { ... }
```

```css path=src/styles.css
...
```

- `src/App.jsx` default-exports a React component. React 18 is available; no other packages.
- `src/styles.css` holds all styling. No external stylesheets, no CDN scripts, no inline `<script>`.
- Assets are referenced by their given relative paths under `/assets/` (copied in by the harness), never fetched.
- The page must render without console errors at 1440, 768 and 390 px widths.
