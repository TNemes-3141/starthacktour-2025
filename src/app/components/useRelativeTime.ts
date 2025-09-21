"use client";

import { useEffect, useState } from "react";

export function useRelativeTime(date: Date, intervalMs = 1000): string {
    const [text, setText] = useState(() => formatRelative(date));

    useEffect(() => {
        const id = setInterval(() => {
            setText(formatRelative(date));
        }, intervalMs);
        return () => clearInterval(id);
    }, [date, intervalMs]);

    return text;
}

function formatRelative(date: Date): string {
    const diffSec = Math.floor((Date.now() - date.getTime()) / 1000);
    if (diffSec < 5) {
        return "just now";
    }
    if (diffSec < 60) {
        return `${diffSec} seconds ago`;
    }
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) {
        return `${diffMin} minutes ago`;
    }
    const diffH = Math.floor(diffMin / 60);
    return `${diffH} hours ago`;
}
