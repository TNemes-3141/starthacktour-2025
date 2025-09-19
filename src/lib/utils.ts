import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

/* Function that can be used to merge Tailwind classnames from variables with static classnames, even conditionally if needed
Example: <body className={cn("flex flex-col min-h-screen bg-background antialiased", primary_font.className)}>
*/
export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
  }