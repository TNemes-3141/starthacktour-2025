"use client";

import DashboardLayout from "./components/DashboardLayout";
import ObjectList from "./components/ObjectList";
import MapView from "./components/MapView";
import { useEffect } from "react";
import { createClient } from "@/lib/supabase/client";
import { useObjectManager } from "./useObjectManager";

export default function Home() {
  const supabase = createClient();
  const { active, past, addOrUpdate } = useObjectManager();

  useEffect(() => {
    // Subscribe to new rows
    const channel = supabase
      .channel("object_updates")
      .on(
        "postgres_changes",
        { event: "INSERT", schema: "public", table: "object_updates" },
        (payload) => {
          addOrUpdate(payload.new as any);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [addOrUpdate]);

  return (
    <DashboardLayout
      leftTop={<ObjectList title="Active Objects" type="active" objects={active} />}
      leftBottom={<ObjectList type="past" objects={past} />}
      right={<MapView />}
    />
  );
}
