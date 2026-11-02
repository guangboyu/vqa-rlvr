/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      // A light, paper-like palette shared with the report page; the method colors live in
      // src/components/ui.tsx because they are picked per model at run time.
      colors: {
        paper: '#ffffff',
        wash: '#f6f6f4',
        rule: '#e6e6e3',
        line: '#cfcfca',
        ink: { DEFAULT: '#1c1d20', 2: '#4a4c52', 3: '#7a7d85', 4: '#a9abb0' },
        accent: '#23466f',
        miss: '#a33a3a',
      },
      fontFamily: {
        serif: ['Newsreader', '"Source Serif 4"', 'Georgia', 'serif'],
        sans: ['"IBM Plex Sans"', '"Helvetica Neue"', 'Arial', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
