import type { Metadata } from "next";
import { primaryFont, monoFont } from "@/lib/fonts";
import "./globals.css";
import { Providers } from "./providers";
import Header from "./components/Header";

export const metadata: Metadata = {
  title: "FlySafe Dashboard",
  description: "Monitoring dashboard for FlySafe",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="light">
      <body className={`${primaryFont.className} antialiased`}>
        <Providers>
          {/* App shell */}
          <div className="min-h-screen flex flex-col">
            <Header imageSrc="/jedsy.png" size={120}/>
            <main className="flex-1 overflow-hidden">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
