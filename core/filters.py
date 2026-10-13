from typing import Any, Dict, List, Optional


def _meta(c: Dict[str, Any]) -> Dict[str, Any]:
    return c.get("metadata", c)


def filter_by_skills(candidates: List[Dict[str, Any]], required_skills: List[str]) -> List[Dict[str, Any]]:
    """Keep candidates having ALL required skills (case-insensitive). Checks the parsed skill list,
    then falls back to the resume text, so a skill mentioned only in experience still counts."""
    required = [s.strip().lower() for s in required_skills if s.strip()]
    if not required:
        return candidates
    out = []
    for c in candidates:
        skills = [str(s).lower() for s in _meta(c).get("skills", [])]
        text = (c.get("text") or "").lower()
        if all(any(req in s for s in skills) or req in text for req in required):
            out.append(c)
    return out


def filter_by_location(candidates: List[Dict[str, Any]], target_locations: List[str]) -> List[Dict[str, Any]]:
    targets = [t.strip().lower() for t in target_locations if t.strip()]
    if not targets:
        return candidates
    return [c for c in candidates
            if any(t in str(_meta(c).get("location") or "").lower() for t in targets)]


def filter_by_experience(candidates: List[Dict[str, Any]], min_experience: float) -> List[Dict[str, Any]]:
    out = []
    for c in candidates:
        exp = _meta(c).get("experience")
        if exp is not None and exp >= min_experience:
            out.append(c)
    return out


def apply_filters(candidates: List[Dict[str, Any]], required_skills: Optional[List[str]] = None,
                  target_locations: Optional[List[str]] = None,
                  min_experience: Optional[float] = None) -> List[Dict[str, Any]]:
    result = list(candidates)
    if required_skills:
        result = filter_by_skills(result, required_skills)
    if target_locations:
        result = filter_by_location(result, target_locations)
    if min_experience is not None:
        result = filter_by_experience(result, min_experience)
    return result