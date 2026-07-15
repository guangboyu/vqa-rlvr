/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#0b0e13', 900: '#10141b', 800: '#171c26', 700: '#222936',
          600: '#2e3745', 400: '#5c6b7f', 300: '#8b99ab', 100: '#dbe2ea', 50: '#f2f5f8',
        },
        ember: { 600: '#c9882a', 500: '#e8a33d', 400: '#f0b65e' },
        verify: '#4cc38a',
      },
      fontFamily: {
        display: ['Georgia', 'Cambria', 'serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
