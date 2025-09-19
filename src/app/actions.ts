"use server"

import { eq } from 'drizzle-orm'

import { createClient } from '@/lib/supabase/server'
import { db } from '@/lib/database/db'
import { usersTable } from '@/lib/database/schema'

export type UserData = {
    id: number,
    username: string,
    age: number,
    email: string,
}

export async function listAllUsers(): Promise<UserData[] | undefined> {
    try {
        const data = await db.select({
            id: usersTable.id,
            username: usersTable.name,
            age: usersTable.age,
            email: usersTable.email,
        }).from(usersTable);
        
        return data;
    } catch (e) {
        console.log("Error while fetching");
    }
}