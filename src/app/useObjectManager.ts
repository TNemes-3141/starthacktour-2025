"use client";

import { useState, useRef } from "react";
import { ObjectUpdate } from "@/lib/database/schema";

export type ManagedObject = ObjectUpdate;

type TimerId = ReturnType<typeof setTimeout>; // works in browser & node typings

export function useObjectManager() {
  const [active, setActive] = useState<ManagedObject[]>([]);
  const [past, setPast] = useState<ManagedObject[]>([]);
  const timers = useRef<Map<string, TimerId>>(new Map());

  function addOrUpdate(obj: ObjectUpdate) {
    // 1) If this object exists in PAST, remove it from PAST first
    setPast((prev) => prev.filter((o) => o.objectId !== obj.objectId));

    // 2) Upsert into ACTIVE (no duplicates)
    setActive((prev) => {
      const idx = prev.findIndex((o) => o.objectId === obj.objectId);
      const next = [...prev];
      if (idx >= 0) {
        next[idx] = { ...next[idx], ...obj }; // update in place
      } else {
        next.push(obj); // insert new
      }
      return next;
    });

    // 3) Reset the 1-minute inactivity timer
    resetTimer(obj.objectId);
  }

  function resetTimer(objectId: string) {
    // Clear old timer (if any)
    const old = timers.current.get(objectId);
    if (old) clearTimeout(old);

    const t = setTimeout(() => {
      // Move from ACTIVE -> PAST exactly once
      setActive((prev) => {
        const obj = prev.find((o) => o.objectId === objectId);
        // If it’s no longer active (e.g., already moved), do nothing
        if (!obj) return prev;

        // Add to PAST without duplicates
        setPast((p) => {
          const without = p.filter((o) => o.objectId !== objectId);
          return [...without, obj];
        });

        // Remove from ACTIVE
        return prev.filter((o) => o.objectId !== objectId);
      });

      timers.current.delete(objectId);
    }, 60_000);

    timers.current.set(objectId, t);
  }

  return { active, past, addOrUpdate };
}
