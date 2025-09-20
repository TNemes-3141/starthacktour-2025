"use client";

import { useEffect, useState } from "react";
import EmptyState from "./EmptyState";
import { ObjectData } from "./types";

interface Props {
  title: string;
  type: "active" | "past";
}

export default function ObjectList({ title, type }: Props) {
  const [objects, setObjects] = useState<ObjectData[]>([]);

  /*useEffect(() => {
    fetchLocations().then((data) => {
      const filtered = data.filter((obj) =>
        type === "active" ? isActive(obj.timestamp) : !isActive(obj.timestamp)
      );
      setObjects(filtered);
    });
  }, [type]);*/

  return (
    <div className="flex flex-col h-full">
      <div className="border-b-1.5 border-b-gray-200 px-4 py-2 text-lg">{title}</div>
      <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-white">
        {objects.length === 0 ? (
          <EmptyState text={`No ${type} objects yet.`} />
        ) : (
          objects.map((obj) => (
            <div
              key={obj.id}
              className={`border rounded p-2 ${
                type === "active" ? "border-green-500" : "border-gray-400"
              }`}
            >
              <div className="font-medium">{obj.name}</div>
              <div className="text-xs text-gray-500">
                {obj.latitude.toFixed(4)}, {obj.longitude.toFixed(4)}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
