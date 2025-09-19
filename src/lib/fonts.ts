import { Geist, Geist_Mono } from "next/font/google";

export const primaryFont = Geist({
    variable: "--font-primary",
    subsets: ["latin"],
});

export const monoFont = Geist_Mono({
    variable: "--font-monospace",
    subsets: ["latin"],
});