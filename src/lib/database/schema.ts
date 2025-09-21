import { pgTable, text, timestamp, doublePrecision, primaryKey } from 'drizzle-orm/pg-core';


export const objectUpdates = pgTable(
    "object_updates",
    {
        objectId: text("objectId").notNull(),
        timestamp: timestamp("timestamp", {
            withTimezone: true,
            precision: 3,
        }).notNull(),
        bboxTop: doublePrecision("bboxTop").notNull(),
        bboxLeft: doublePrecision("bboxLeft").notNull(),
        bboxBottom: doublePrecision("bboxBottom").notNull(),
        bboxRight: doublePrecision("bboxRight").notNull(),
        class: text("class").notNull(),
        confidence: doublePrecision("confidence").notNull(),
        speedMps: doublePrecision("speedMps"),
        distanceM: doublePrecision("distanceM"),
        latitude: doublePrecision("latitude"),
        longitude: doublePrecision("longitude"),
        snapshotPath: text("snapshotPath").notNull(),
        snapshotUrl: text("snapshotUrl"),
    },
    (table) => [
        primaryKey({ columns: [table.objectId, table.timestamp] }),
    ]
);

export type ObjectUpdate = typeof objectUpdates.$inferSelect;
