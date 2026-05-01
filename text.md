# Frontend Usage Prompt — User Routes (In‑Depth)

You are building a frontend for the Plant Disease backend. Implement user profile CRUD using these routes:

**Routes**
- `POST /users` (multipart/form-data): create user profile
- `GET /users/{uid}`: fetch user profile
- `PATCH /users/{uid}` (multipart/form-data): update user profile
- `POST /auth/register` (JSON): firebase-only register

Use `photo_url` returned from `/users` endpoints to show the profile image. If a photo is uploaded, the backend uses the existing upload pipeline and returns a public URL (`photo_url`) plus the storage key (`photo_object_name`).

---

## 1) Data Model (Frontend Types)

```ts
export type UserProfile = {
  id: string;
  firebase_uid: string;
  email: string | null;
  name: string | null;
  photo_object_name: string | null;
  photo_url: string | null;
  phone_number: string | null;
  years_of_experience: number | null;
  acres: number | null;
  primary_crops: string[] | null;
  soil_type: string | null;
};
```

---

## 2) Create Profile (POST /users)

### When to call
After Firebase auth is complete and you have a `uid`. Use this only once per user. If the user already exists, the API returns **409**.

### FormData (multipart)
- `uid` (required)
- `name` (optional)
- `email` (optional)
- `phone_number` (optional)
- `years_of_experience` (optional, integer)
- `acres` (optional, float)
- `primary_crops` (optional, comma-separated: "rice,wheat,corn")
- `soil_type` (optional)
- `photo` (optional file)

### Fetch example
```ts
async function createProfile({
  uid,
  name,
  email,
  phoneNumber,
  yearsOfExperience,
  acres,
  primaryCrops,
  soilType,
  photoFile,
}: {
  uid: string;
  name?: string;
  email?: string;
  phoneNumber?: string;
  yearsOfExperience?: number;
  acres?: number;
  primaryCrops?: string[];
  soilType?: string;
  photoFile?: File;
}): Promise<UserProfile> {
  const form = new FormData();
  form.append("uid", uid);
  if (name) form.append("name", name);
  if (email) form.append("email", email);
  if (phoneNumber) form.append("phone_number", phoneNumber);
  if (yearsOfExperience !== undefined) form.append("years_of_experience", yearsOfExperience.toString());
  if (acres !== undefined) form.append("acres", acres.toString());
  if (primaryCrops?.length) form.append("primary_crops", primaryCrops.join(","));
  if (soilType) form.append("soil_type", soilType);
  if (photoFile) form.append("photo", photoFile);

  const res = await fetch(`${API_BASE}/users`, {
    method: "POST",
    body: form,
  });

  if (res.status === 409) {
    throw new Error("User already exists");
  }
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}
```

---

## 3) Read Profile (GET /users/{uid})

### When to call
On app load, after login, or when rendering a profile screen.

```ts
async function getProfile(uid: string): Promise<UserProfile> {
  const res = await fetch(`${API_BASE}/users/${uid}`);
  if (res.status === 404) throw new Error("User not found");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

---

## 4) Update Profile (PATCH /users/{uid})

### When to call
From profile edit screen. Only send fields that changed. If you pass an email that already exists, the API returns **409**.

### FormData (multipart)
- `name` (optional)
- `email` (optional)
- `phone_number` (optional)
- `years_of_experience` (optional, integer)
- `acres` (optional, float)
- `primary_crops` (optional, comma-separated)
- `soil_type` (optional)
- `photo` (optional file)

```ts
async function updateProfile(
  uid: string,
  updates: {
    name?: string;
    email?: string;
    phoneNumber?: string;
    yearsOfExperience?: number;
    acres?: number;
    primaryCrops?: string[];
    soilType?: string;
    photoFile?: File;
  }
): Promise<UserProfile> {
  const form = new FormData();
  if (updates.name !== undefined) form.append("name", updates.name);
  if (updates.email !== undefined) form.append("email", updates.email);
  if (updates.phoneNumber !== undefined) form.append("phone_number", updates.phoneNumber);
  if (updates.yearsOfExperience !== undefined) form.append("years_of_experience", updates.yearsOfExperience.toString());
  if (updates.acres !== undefined) form.append("acres", updates.acres.toString());
  if (updates.primaryCrops?.length) form.append("primary_crops", updates.primaryCrops.join(","));
  if (updates.soilType !== undefined) form.append("soil_type", updates.soilType);
  if (updates.photoFile) form.append("photo", updates.photoFile);

  const res = await fetch(`${API_BASE}/users/${uid}`, {
    method: "PATCH",
    body: form,
  });

  if (res.status === 409) throw new Error("Email already registered");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

---

## 5) Firebase-only Register (POST /auth/register)

Use this only if you want a lightweight user creation with no profile info. This does **not** upload images.

```ts
async function registerFirebaseUser(uid: string, email?: string) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ firebase_uid: uid, email }),
  });
  if (res.status === 409) throw new Error("Email already registered");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

---

## 6) UI Flow (Recommended)

1. User logs in via Firebase → you get `uid` and maybe `email`.
2. Call `GET /users/{uid}`.
   - If **404**, call `POST /users` to create profile.
3. Store the profile in state.
4. Render profile screen with form for all farmer fields.
5. On profile edits, call `PATCH /users/{uid}` with only changed fields.

---

## 7) Handling Errors

- **409 Conflict**: user exists or email already registered. Show a message to user.
- **404 Not Found**: user does not exist (create profile).
- **415 Unsupported Media Type**: uploaded file is not an image.
- **400 Bad Request**: missing `uid`, no update fields, or invalid data.

---

## 8) Rendering Profile Photo

If `photo_url` is present:
```tsx
<img src={profile.photo_url} alt="Profile" />
```

If it is null, render a placeholder avatar.

---

## 9) Farmer Profile Form Example (React)

```tsx
import { useState } from "react";

export function FarmerProfileForm({ uid }: { uid: string }) {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [form, setForm] = useState({
    name: "",
    email: "",
    phoneNumber: "",
    yearsOfExperience: "",
    acres: "",
    primaryCrops: "",
    soilType: "",
    photo: null as File | null,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!profile) {
      await createProfile({
        uid,
        ...form,
        yearsOfExperience: form.yearsOfExperience ? parseInt(form.yearsOfExperience) : undefined,
        acres: form.acres ? parseFloat(form.acres) : undefined,
        primaryCrops: form.primaryCrops ? form.primaryCrops.split(",").map(c => c.trim()) : undefined,
        photoFile: form.photo || undefined,
      });
    } else {
      await updateProfile(uid, {
        ...form,
        yearsOfExperience: form.yearsOfExperience ? parseInt(form.yearsOfExperience) : undefined,
        acres: form.acres ? parseFloat(form.acres) : undefined,
        primaryCrops: form.primaryCrops ? form.primaryCrops.split(",").map(c => c.trim()) : undefined,
        photoFile: form.photo || undefined,
      });
    }
    const updated = await getProfile(uid);
    setProfile(updated);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Name"
        value={form.name}
        onChange={(e) => setForm({ ...form, name: e.target.value })}
      />
      <input
        type="email"
        placeholder="Email"
        value={form.email}
        onChange={(e) => setForm({ ...form, email: e.target.value })}
      />
      <input
        type="tel"
        placeholder="Phone"
        value={form.phoneNumber}
        onChange={(e) => setForm({ ...form, phoneNumber: e.target.value })}
      />
      <input
        type="number"
        placeholder="Years of Experience"
        value={form.yearsOfExperience}
        onChange={(e) => setForm({ ...form, yearsOfExperience: e.target.value })}
      />
      <input
        type="number"
        placeholder="Acres"
        step="0.1"
        value={form.acres}
        onChange={(e) => setForm({ ...form, acres: e.target.value })}
      />
      <input
        type="text"
        placeholder="Primary Crops (comma-separated)"
        value={form.primaryCrops}
        onChange={(e) => setForm({ ...form, primaryCrops: e.target.value })}
      />
      <input
        type="text"
        placeholder="Soil Type"
        value={form.soilType}
        onChange={(e) => setForm({ ...form, soilType: e.target.value })}
      />
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setForm({ ...form, photo: e.target.files?.[0] || null })}
      />
      <button type="submit">{profile ? "Update" : "Create"} Profile</button>
    </form>
  );
}
```

---

## 10) Notes for Frontend Implementation

- Use `multipart/form-data` for any request that includes `photo`.
- Don't set `Content-Type` manually for `FormData` requests; the browser sets boundaries.
- Store `firebase_uid` in your auth state. It is required for all `/users` requests.
- `primary_crops` is sent as comma-separated string, returned as JSON array of strings.
- If you have a backend base URL mismatch, configure `API_BASE` once (e.g. `.env`).
