import { FaRobot } from "react-icons/fa";
import { FaUser } from "react-icons/fa";

export default function ChatAvatar({ role }: { role: string }) {
  if (role === "user") {
    return (
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border bg-background shadow">
        <FaUser className="h-4 w-4" />
      </div>
    );
  }

  return (
    <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border bg-background shadow">
      <FaRobot className="h-4 w-4" />
    </div>
  );
}
