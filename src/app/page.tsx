"use client";
import { Button, ButtonGroup } from "@heroui/button";
import { listAllUsers, UserData } from "./actions";

export default function Home() {
  var users: UserData[] | undefined;

  var getUsers = async () => {
    users = await listAllUsers();
    console.log(users);
  }

  return (
    <div className="font-sans grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        <Button color="primary" onPress={getUsers}>Button</Button>
      </main>
    </div>
  );
}
