/** The languages you can dictate in, by the code the backend stores. */
export const DICTATION_LANGUAGES = [
  { code: "uk", label: "Ukrainian" },
  { code: "en", label: "English" },
  { code: "de", label: "German" },
  { code: "fr", label: "French" },
  { code: "es", label: "Spanish" },
  { code: "pl", label: "Polish" },
  { code: "ja", label: "Japanese" },
  { code: "zh", label: "Chinese" },
] as const;

export function languageName(code: string): string {
  return DICTATION_LANGUAGES.find((language) => language.code === code)?.label ?? code;
}
