"use client";

import EmptyState from "./EmptyState";
import { ManagedObject } from "../useObjectManager";
import ObjectCard from "./ObjectCard";

interface Props {
  title?: string;
  type: "active" | "past";
  objects: ManagedObject[];
}

export default function ObjectList({ title, type, objects }: Props) {
  return (
    <div className="flex flex-col h-full">
      {title && (
        <div className="border-b-1.5 border-b-gray-200 px-4 py-2 text-lg">
          {title}
        </div>
      )}
      <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-white">
        {objects.length === 0 ? (
          <EmptyState text={`No ${type} objects yet.`} />
        ) : (
          objects.map((obj) => (
            <ObjectCard
              key={obj.objectId}
              obj={obj}
              variant={type === "active" ? "relative" : "static"}
            />
          ))
        )}
      </div>
    </div>
  );
}
