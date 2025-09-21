import { Spinner } from "@heroui/react";

interface Props {
    text: string;
}

export default function EmptyState({ text }: Props) {
    return (
        <div className="flex gap-4 items-center justify-center h-full text-gray-400 italic">
            <Spinner variant="dots" color="default" />
            <span>{text}</span>
        </div>
    );
}
