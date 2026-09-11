/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', '"Courier New"', 'monospace'],
        display: ['"Inter"', 'system-ui', 'sans-serif'],
      },
      colors: {
        ocean: {
          950: '#010b1a',
          900: '#020d1f',
          800: '#061529',
          700: '#0a1e35',
          600: '#0e2845',
          500: '#133458',
        },
        radar: {
          500: '#00ff88',
          400: '#39ffb0',
          300: '#7affcc',
          200: '#b3ffe3',
          glow: 'rgba(0,255,136,0.15)',
        },
        sonar: {
          500: '#00d4ff',
          400: '#33ddff',
          300: '#66e5ff',
          glow: 'rgba(0,212,255,0.15)',
        },
        hazard: {
          500: '#ff6b00',
          400: '#ff8c33',
          300: '#ffad66',
          glow: 'rgba(255,107,0,0.2)',
        },
        caution: {
          500: '#ffd700',
          400: '#ffe033',
        },
      },
      boxShadow: {
        'radar-glow':  '0 0 20px rgba(0,255,136,0.3), 0 0 40px rgba(0,255,136,0.1)',
        'sonar-glow':  '0 0 20px rgba(0,212,255,0.3), 0 0 40px rgba(0,212,255,0.1)',
        'hazard-glow': '0 0 20px rgba(255,107,0,0.35)',
        'panel':       '0 0 0 1px rgba(0,212,255,0.1), 0 8px 32px rgba(0,0,0,0.6)',
        'inset-panel': 'inset 0 1px 0 rgba(0,212,255,0.1)',
      },
      animation: {
        'pulse-radar': 'pulse-radar 2s ease-in-out infinite',
        'scan-line':   'scan-line 3s linear infinite',
        'blink':       'blink 1.2s step-end infinite',
        'spin-slow':   'spin 3s linear infinite',
      },
      keyframes: {
        'pulse-radar': {
          '0%, 100%': { opacity: 1 },
          '50%':       { opacity: 0.4 },
        },
        'scan-line': {
          '0%':   { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        'blink': {
          '0%, 100%': { opacity: 1 },
          '50%':      { opacity: 0 },
        },
      },
    },
  },
  plugins: [],
}
