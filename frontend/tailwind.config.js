/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        editorial: ["Times New Roman", "Times", "serif"]
      }
    }
  },
  plugins: []
};
