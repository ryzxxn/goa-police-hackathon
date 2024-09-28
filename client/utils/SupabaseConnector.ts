import { SupabaseClient } from "@supabase/supabase-js";

const URL = process.env.SUPABASE_URL as string;
const KEY = process.env.SUPABASE_KEY as string;

export const Supabase = new SupabaseClient(URL, KEY)