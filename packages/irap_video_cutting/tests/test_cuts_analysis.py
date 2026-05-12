import json
import os

# Load original JSON
original = json.load(open('input_webgis/VID_20250104_085135_00_008_20251122224908.json'))
points = [p for p in original if 'time' in p]

print("=== Original video duration and key timestamps ===")
print(f"First point: time={points[0]['time']}, coords={points[0]['coordinates']}")
print(f"Last point: time={points[-1]['time']}, coords={points[-1]['coordinates']}")

print("\n=== Test Cut: 0-20s ===")
cut1 = json.load(open('output_webgis_test/VID_20250104_085135_00_008_20251122224908_1.json'))
cut1_points = [p for p in cut1 if 'time' in p]
print(f"First point: time={cut1_points[0]['time']}, coords={cut1_points[0]['coordinates']}")
print(f"Last point: time={cut1_points[-1]['time']}, coords={cut1_points[-1]['coordinates']}")

print("\n=== Test Cut: 20-40s (re-zeroed) ===")
cut2 = json.load(open('output_webgis_test/VID_20250104_085135_00_008_20251122224908_2.json'))
cut2_points = [p for p in cut2 if 'time' in p]
print(f"First point: time={cut2_points[0]['time']}, coords={cut2_points[0]['coordinates']}")
print(f"Last point: time={cut2_points[-1]['time']}, coords={cut2_points[-1]['coordinates']}")

print("\n=== Test Cut: 40-60s (re-zeroed) ===")
cut3 = json.load(open('output_webgis_test/VID_20250104_085135_00_008_20251122224908_3.json'))
cut3_points = [p for p in cut3 if 'time' in p]
print(f"First point: time={cut3_points[0]['time']}, coords={cut3_points[0]['coordinates']}")
print(f"Last point: time={cut3_points[-1]['time']}, coords={cut3_points[-1]['coordinates']}")

print("\n=== Test Cut: 60-95.7s (re-zeroed) ===")
cut4 = json.load(open('output_webgis_test/VID_20250104_085135_00_008_20251122224908_4.json'))
cut4_points = [p for p in cut4 if 'time' in p]
print(f"First point: time={cut4_points[0]['time']}, coords={cut4_points[0]['coordinates']}")
print(f"Last point: time={cut4_points[-1]['time']}, coords={cut4_points[-1]['coordinates']}")

print("\n=== Checking SQL boundary coordinates ===")
# Extract SQL linestring to count coordinates
with open('output_webgis_test/VID_20250104_085135_00_008_20251122224908_1.sql') as f:
    sql = f.read()
    # Count coordinates in LINESTRING
    linestring_start = sql.find('LINESTRING(')
    linestring_end = sql.find(')', linestring_start)
    linestring_content = sql[linestring_start+11:linestring_end]
    coords = linestring_content.split(',')
    print(f"Segment 1 SQL has {len(coords)} coordinate pairs")
    print(f"First: {coords[0]}")
    print(f"Last: {coords[-1]}")
