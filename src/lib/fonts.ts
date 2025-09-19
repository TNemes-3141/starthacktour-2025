import { Geist, Geist_Mono } from "next/font/google";

export const primaryFont = Geist({
    variable: "--font-geist-sans",
    subsets: ["latin"],
});

export const secondaryFont = Geist_Mono({
    variable: "--font-geist-mono",
    subsets: ["latin"],
});