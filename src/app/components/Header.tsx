"use client";

import Image from "next/image";
import { cn } from "@/lib/utils"; // optional helper if you have one; otherwise remove
import { Navbar, NavbarBrand, NavbarContent } from "@heroui/react";

type HeaderProps = {
  /** Optional PNG/SVG path. If omitted, a gray placeholder is shown. */
  imageSrc?: string;
  imageAlt?: string;
  /** Size of the overhanging image/placeholder */
  size?: number; // px
};

export default function Header({
  imageSrc,
  imageAlt = "FlySafe decorative image",
  size = 64,
}: HeaderProps) {
  return (
    <Navbar position="static" className="bg-green-700">
      <NavbarBrand>
        <span className="text-2xl text-white font-semibold tracking-wide">FlySafe</span>
        {imageSrc ? (
          <Image
            src={imageSrc}
            alt={imageAlt}
            width={size}
            height={size}
            className="object-contain drop-shadow-sm pointer-events-none select-none"
            priority
          />
        ) : (
          <div
            aria-hidden
            className={cn(
              "w-full h-full rounded-md bg-gray-300/80 dark:bg-gray-600/80",
              "pointer-events-none"
            )}
          />
        )}
      </NavbarBrand>
      <NavbarContent>

      </NavbarContent>
    </Navbar>
  );
}

/*<header className="relative h-12 bg-green-900 text-white px-4 flex items-center">
<span className="text-2xl font-semibold tracking-wide">FlySafe</span>
<div
  className="absolute left-27 -top-5"
  style={{ width: size, height: size }}
>
  {imageSrc ? (
    <Image
      src={imageSrc}
      alt={imageAlt}
      width={size}
      height={size}
      className="object-contain drop-shadow-sm pointer-events-none select-none"
      priority
    />
  ) : (
    <div
      aria-hidden
      className={cn(
        "w-full h-full rounded-md bg-gray-300/80 dark:bg-gray-600/80",
        "pointer-events-none"
      )}
    />
  )}
</div>
</header>*/