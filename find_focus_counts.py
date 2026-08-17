"""
Interactive focus-count finder: drive the emulated focus while watching the
lens distance bar on the camera screen, and note the count at each product
distance. Counts found here are set_focus_position / focus_on_connect values.

Run from ~/focus on the machine:  python3 find_focus_counts.py

Commands (position = emulated nudge count from the near stop):
    +N        move N counts toward far   (e.g. +10)
    -N        move N counts toward near  (e.g. -3)
    g N       go to absolute count N     (e.g. g 88)
    h         re-home (near stop, count 0)
    c         capture a test frame (prints the file path)
    s LABEL   record current count under LABEL (printed at exit)
    q         quit (prints everything recorded)

Derive with the same config playback will use (emulated_step_size 3,
emulated_nudge_interval_s 0.15), lens switch on MF. The scale is nonlinear in
distance - measure every product distance on the bar, don't interpolate.
For a final check on each count: h, then g <count>, then c - the distance bar
and the test frame are the proof.
"""

import asyncio

from viam.robot.client import RobotClient
from viam.components.camera import Camera


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key='9mutj1rhbn3oypsmohhzoci3bqxbuw9f',
        api_key_id='eef423e7-6863-4920-9c07-a514b833bae5'
    )
    return await RobotClient.at_address('nines-photographer.g02wbvukyo.viam.cloud', opts)


async def main():
    async with await connect() as machine:
        sony = Camera.from_robot(machine, "sony")
        saved = {}

        homed = await sony.do_command({"command": "home_focus"})
        position = int(homed.get("position", 0))
        print(f"homed: position {position}. Watch the distance bar on the "
              f"camera screen; type commands (+N / -N / g N / h / c / s LABEL / q).")

        while True:
            try:
                line = input(f"[{position:>4}] > ").strip()
            except EOFError:
                break
            if not line:
                continue

            try:
                if line == "q":
                    break
                elif line == "h":
                    await sony.do_command({"command": "home_focus"})
                    position = 0
                    print("homed (count 0, near stop)")
                elif line == "c":
                    cap = (await sony.do_command({"capture": {}}))["capture"]
                    print(f"captured {cap.get('name')} -> {cap.get('saved_to')}")
                elif line.startswith("s "):
                    label = line[2:].strip()
                    saved[label] = position
                    print(f"recorded {label} = {position}")
                elif line.startswith("g "):
                    target = int(line[2:])
                    resp = await sony.do_command(
                        {"command": "set_focus_position", "position": target})
                    position = int(resp.get("position", target))
                    print(f"at {position}")
                elif line[0] in "+-":
                    delta = int(line)
                    resp = await sony.do_command(
                        {"command": "set_focus_position", "position": position + delta})
                    position = int(resp.get("position", position + delta))
                    print(f"at {position}")
                else:
                    print("commands: +N  -N  g N  h  c  s LABEL  q")
            except Exception as exc:
                print(f"error: {exc}")

        if saved:
            print("\nrecorded counts (set_focus_position / focus_on_connect values):")
            for label, count in saved.items():
                print(f"  {label}: {count}")


if __name__ == '__main__':
    asyncio.run(main())
