// lib/actions.ts
"use server";

import { db } from "@/lib/database/db"; // your Drizzle db instance
import { objectUpdates, ObjectUpdate } from "@/lib/database/schema";
import { desc, sql } from "drizzle-orm";

/**
 * Get the latest row per object_id
 */
export async function listLatestObjects() {
  return db.execute<ObjectUpdate>(
    sql`
      SELECT DISTINCT ON (object_id) *
      FROM ${objectUpdates}
      ORDER BY object_id, "timestamp" DESC;
    `
  );
}
