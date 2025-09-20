"use client";

import { useState, useMemo } from "react";
import { Button } from "@heroui/button";
import { ChevronUp, ChevronDown } from "lucide-react";

type Props = {
    title: string;
    /** Height when opened. Can be a CSS length (e.g. "60%", "320px"). */
    expandedHeight?: string;
    defaultOpen?: boolean;
    children: React.ReactNode;
};

/**
 * Collapsed: only a sticky header (about 44px tall) visible at the bottom.
 * Expanded: panel grows from bottom to top and reveals its children.
 */
export default function ExpandableBottomPanel({
    title,
    expandedHeight = "55%",
    defaultOpen = false,
    children,
}: Props) {
    const [open, setOpen] = useState(defaultOpen);

    // 44px header (h-11) when collapsed; otherwise use the configured height
    const containerStyle = useMemo<React.CSSProperties>(
        () => ({
            height: open ? expandedHeight : "44px",
            transition: "height 220ms ease",
        }),
        [open, expandedHeight]
    );

    return (
        <div
            className="border-t border-gray-200 bg-white rounded-t-md shadow-sm overflow-hidden"
            style={containerStyle}
        >
            {/* Header bar (always visible, sticks to bottom of left column) */}
            <div className="h-11 flex items-center px-3">
                <div className="flex-1 text-lg font-medium text-gray-700 select-none">
                    {title}
                </div>
                <Button
                    size="sm"
                    variant="light"
                    isIconOnly
                    aria-label={open ? "Collapse" : "Expand"}
                    onPress={() => setOpen((v) => !v)}
                >
                    {open ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                </Button>
            </div>

            {/* Content (only rendered when open to avoid useless layout work) */}
            {open && (
                <div className="h-[calc(100%-44px)] overflow-auto p-3">
                    {children}
                </div>
            )}
        </div>
    );
}
