/** @type {import('tailwindcss').Config} */
export default {
  content: ["./views/**/*.ejs", "./public/**/*.js"],
  theme: {
    extend: {
      colors: {
        brand: "#F6DE82",
        "brand-soft": "#FBEFC0",
        "brand-strong": "#DCB210",
        ink: "#1A1A1A",
        "ink-dark": "#232A31",
        "ink-soft": "#3C4854",
        "ink-muted": "#5A6D7F",
        success: "#1FC16B",
        "success-light": "#84EBB4",
      },
      fontFamily: {
        sans: ["Helvetica", "Arial", "sans-serif"],
      },
      letterSpacing: {
        tight: "-0.02em",
        tighter: "-0.04em",
      },
    },
  },
  plugins: [],
};
