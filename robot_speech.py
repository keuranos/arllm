"""
Robot Speech & Commentary System

Generates Finnish speech/commentary for the robot using:
1. LLM for generating contextual Finnish phrases
2. Android TTS endpoint for speech synthesis

The robot comments on what it's doing, reacts to observations,
and can have simple conversations.
"""

import random
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import requests
import json


# ========================
# Configuration
# ========================

# Try to import config, fallback to defaults
try:
    from config import Config
    OLLAMA_BASE = Config.OLLAMA_BASE
    SPEECH_MODEL = Config.MODEL
    GATEWAY_BASE = Config.GATEWAY_BASE
except ImportError:
    OLLAMA_BASE = "http://192.168.86.250:11434"
    SPEECH_MODEL = "devstral-small-2:latest"
    GATEWAY_BASE = "http://192.168.86.35:8765"

SESSION = requests.Session()


# ========================
# Commentary Types
# ========================

class CommentaryType(Enum):
    ACTION = "action"           # Commenting on what robot is doing
    OBSERVATION = "observation" # Describing what robot sees
    REACTION = "reaction"       # Emotional/reactive comment
    GREETING = "greeting"       # Social interaction
    THINKING = "thinking"       # "Hmm, I wonder..."
    NARRATION = "narration"     # General narration


# ========================
# Pre-defined Finnish Phrases (fallback)
# Doctor personality - professional, caring, observant
# ========================

FINNISH_PHRASES = {
    "exploring": [
        "Suoritan tarkastuskierroksen...",
        "Tarkistan tilanteen.",
        "Teen havaintoja ympäristöstä.",
        "Jatkan tutkimusta.",
    ],
    "found_person": [
        "Hyvää päivää! Näen potilaan.",
        "Havaitsin henkilön. Kaikki kunnossa?",
        "Siellä on joku. Tarkistan tilanteen.",
        "Tervehdys! Miten voin auttaa?",
    ],
    "approaching": [
        "Tulen lähemmäs tarkastamaan...",
        "Siirryn paikalle.",
        "Lähestyn rauhallisesti.",
    ],
    "turning": [
        "Käännyn tarkistamaan...",
        "Vaihdan näkökulmaa.",
        "Tarkistan sivusuunnan.",
    ],
    "obstacle": [
        "Este havaittu. Kierrän sen.",
        "Täältä ei pääse. Valitsen toisen reitin.",
        "Navigoin esteen ohi.",
    ],
    "curious": [
        "Mielenkiintoinen havainto...",
        "Tämä vaatii lähempää tarkastelua.",
        "Analysoin tilannetta.",
        "Dokumentoin löydöksen.",
    ],
    "bored": [
        "Odottelen seuraavia ohjeita.",
        "Tilanne rauhallinen. Jatkan valvontaa.",
        "Päivystän täällä.",
    ],
    "found_door": [
        "Kulkureitti havaittu.",
        "Tästä pääsee toiseen tilaan.",
        "Oviaukko tunnistettu.",
    ],
    "entering_room": [
        "Siirryn seuraavaan tilaan...",
        "Aloitan uuden tilan tarkastuksen.",
        "Teen kierroksen täällä.",
    ],
    "greeting": [
        "Hyvää päivää!",
        "Tervehdys!",
        "Tohtori Ramble palveluksessanne.",
        "Päivää päivää!",
    ],
    "thinking": [
        "Analysoidaan...",
        "Pohditaanpa...",
        "Hetkinen, prosessoin...",
        "Arvioin tilannetta...",
    ],
    "success": [
        "Tarkastus suoritettu.",
        "Tehtävä valmis.",
        "Homma hoidettu.",
        "Tehty!",
    ],
    "confused": [
        "En ihan ymmärrä...",
        "Missä mä olen?",
        "Hetkinen, mietin...",
    ],
}


# ========================
# TTS Client (Android)
# ========================

class TTSClient:
    """Client for Android TTS endpoint."""

    def __init__(self, gateway_base: str = GATEWAY_BASE):
        self.gateway_base = gateway_base
        self.tts_url = f"{gateway_base}/speak"
        self.enabled = True
        self.last_speak_time = 0
        self.min_interval = 2.0  # Minimum seconds between utterances

    def speak(self, text: str, lang: str = "fi-FI", queue: bool = False) -> bool:
        """
        Send text to Android TTS.

        Args:
            text: Text to speak
            lang: Language code (fi-FI for Finnish)
            queue: If True, queue after current speech; if False, interrupt
        """
        if not self.enabled or not text:
            return False

        # Rate limiting
        now = time.time()
        if now - self.last_speak_time < self.min_interval:
            return False

        try:
            params = {
                "text": text,
                "lang": lang,
                "queue": "1" if queue else "0",
            }
            r = SESSION.get(self.tts_url, params=params, timeout=5)
            self.last_speak_time = now
            return r.status_code == 200
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return False

    def stop(self):
        """Stop current speech."""
        try:
            SESSION.get(f"{self.gateway_base}/speak", params={"stop": "1"}, timeout=2)
        except Exception:
            pass


# ========================
# LLM Commentary Generator
# ========================

class CommentaryGenerator:
    """Generates Finnish commentary using LLM."""

    def __init__(self, ollama_base: str = OLLAMA_BASE, model: str = SPEECH_MODEL):
        self.ollama_base = ollama_base
        self.model = model
        self.last_comments: List[str] = []  # Avoid repetition

    def generate(
        self,
        context: Dict[str, Any],
        commentary_type: CommentaryType = CommentaryType.ACTION,
        max_length: int = 50
    ) -> Optional[str]:
        """
        Generate a Finnish comment based on context.

        Args:
            context: Current situation (action, observations, mood, etc.)
            commentary_type: Type of comment to generate
            max_length: Maximum characters

        Returns:
            Finnish comment string or None
        """
        prompt = self._build_prompt(context, commentary_type, max_length)

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 50,
                }
            }
            r = SESSION.post(f"{self.ollama_base}/api/generate", json=payload, timeout=10)
            r.raise_for_status()
            response = r.json().get("response", "").strip()

            # Clean up response
            response = self._clean_response(response)

            # Avoid repetition
            if response in self.last_comments:
                return None
            self.last_comments.append(response)
            if len(self.last_comments) > 10:
                self.last_comments.pop(0)

            return response

        except Exception as e:
            print(f"[Commentary] LLM error: {e}")
            return None

    def _build_prompt(
        self,
        context: Dict[str, Any],
        commentary_type: CommentaryType,
        max_length: int
    ) -> str:
        """Build prompt for commentary generation."""

        system = """Olet ystävällinen robotti-lemmikki. Puhut suomea.
Tehtäväsi on kommentoida lyhyesti mitä teet tai näet.
Käytä tuttavallista, rentoa kieltä. Voit käyttää huudahduksia ja tunteita.
Vastaa VAIN kommentilla, ei mitään muuta. Max {} merkkiä.
Älä toista samoja fraaseja.""".format(max_length)

        action = context.get("action", "")
        observation = context.get("observation", "")
        mood = context.get("mood", "curious")
        target = context.get("target", "")

        examples = {
            CommentaryType.ACTION: "Esim: 'Lähden tutkimaan!', 'Käännyn oikealle...', 'Menen eteenpäin.'",
            CommentaryType.OBSERVATION: "Esim: 'Oho, joku siellä!', 'Näen oven!', 'Tuo näyttää kiinnostavalta.'",
            CommentaryType.REACTION: "Esim: 'Jee!', 'Oho!', 'Hmm...', 'Hauska nähdä!'",
            CommentaryType.GREETING: "Esim: 'Moi moi!', 'Hei!', 'Terve terve!'",
            CommentaryType.THINKING: "Esim: 'Hmm, mitäköhän...', 'Mietitäänpä...', 'Entäs jos...'",
            CommentaryType.NARRATION: "Esim: 'Tutkin huonetta.', 'Etsin ihmisiä.', 'Kierrän esteettä.'",
        }

        user_prompt = f"""Tyyppi: {commentary_type.value}
Mieliala: {mood}
Toiminto: {action}
Havainto: {observation}
Kohde: {target}

{examples.get(commentary_type, '')}

Kommentti:"""

        return system + "\n\n" + user_prompt

    def _clean_response(self, text: str) -> str:
        """Clean up LLM response."""
        # Remove quotes if present
        text = text.strip().strip('"').strip("'")
        # Remove markdown
        text = text.replace("*", "").replace("_", "")
        # Take first line only
        text = text.split("\n")[0]
        # Limit length
        if len(text) > 100:
            text = text[:97] + "..."
        return text


# ========================
# Robot Personality
# ========================

@dataclass
class RobotPersonality:
    """Robot's speaking personality."""
    name: str = "Robo"
    chattiness: float = 0.5      # 0-1, how often to comment
    enthusiasm: float = 0.7      # 0-1, excitement level
    use_llm: bool = True         # Use LLM or fallback phrases
    language: str = "fi-FI"


class RobotSpeaker:
    """
    Main interface for robot speech.

    Combines LLM generation, TTS playback, and personality.
    """

    def __init__(
        self,
        gateway_base: str = GATEWAY_BASE,
        ollama_base: str = OLLAMA_BASE,
        personality: RobotPersonality = None
    ):
        self.tts = TTSClient(gateway_base)
        self.generator = CommentaryGenerator(ollama_base)
        self.personality = personality or RobotPersonality()
        self.last_speech_time = 0
        self.speech_cooldown = 3.0  # Seconds between speeches

    def maybe_comment(
        self,
        action: str = "",
        observation: str = "",
        mood: str = "neutral",
        target: str = "",
        commentary_type: CommentaryType = CommentaryType.ACTION,
        force: bool = False
    ) -> Optional[str]:
        """
        Maybe generate and speak a comment based on personality.

        Returns the comment if spoken, None otherwise.
        """
        # Check cooldown
        now = time.time()
        if not force and now - self.last_speech_time < self.speech_cooldown:
            return None

        # Check chattiness
        if not force and random.random() > self.personality.chattiness:
            return None

        context = {
            "action": action,
            "observation": observation,
            "mood": mood,
            "target": target,
        }

        # Generate comment
        if self.personality.use_llm:
            comment = self.generator.generate(context, commentary_type)
            if not comment:
                # Fallback to pre-defined phrases
                comment = self._get_fallback_phrase(action, observation, mood)
        else:
            comment = self._get_fallback_phrase(action, observation, mood)

        if comment:
            self.speak(comment)
            return comment

        return None

    def speak(self, text: str, queue: bool = False) -> bool:
        """Speak text via TTS."""
        if not text:
            return False

        print(f"[SPEAK] {text}")
        self.last_speech_time = time.time()
        return self.tts.speak(text, self.personality.language, queue)

    def _get_fallback_phrase(
        self,
        action: str,
        observation: str,
        mood: str
    ) -> Optional[str]:
        """Get a pre-defined Finnish phrase based on context."""
        action_lower = action.lower()
        obs_lower = observation.lower()

        # Match action keywords
        if "explor" in action_lower or "tutkis" in action_lower:
            return random.choice(FINNISH_PHRASES["exploring"])
        if "turn" in action_lower or "käänty" in action_lower:
            return random.choice(FINNISH_PHRASES["turning"])
        if "approach" in action_lower or "lähesty" in action_lower:
            return random.choice(FINNISH_PHRASES["approaching"])
        if "block" in action_lower or "obstacle" in action_lower:
            return random.choice(FINNISH_PHRASES["obstacle"])

        # Match observation keywords
        if "person" in obs_lower or "ihminen" in obs_lower:
            return random.choice(FINNISH_PHRASES["found_person"])
        if "door" in obs_lower or "ovi" in obs_lower or "opening" in obs_lower:
            return random.choice(FINNISH_PHRASES["found_door"])

        # Match mood
        if mood == "curious":
            return random.choice(FINNISH_PHRASES["curious"])
        if mood == "social":
            return random.choice(FINNISH_PHRASES["greeting"])
        if mood == "tired":
            return random.choice(FINNISH_PHRASES["bored"])

        # Default
        return random.choice(FINNISH_PHRASES["thinking"])

    def greet(self):
        """Say a greeting."""
        greetings = [
            "Moi moi! Mitä kuuluu?",
            "Hei hei! Kiva nähdä!",
            "Terve! Onpa mukavaa!",
            "Moro! Mitäs täällä?",
        ]
        self.speak(random.choice(greetings))

    def react_to_person(self, distance: str = "far"):
        """React to seeing a person."""
        if distance == "close":
            phrases = [
                "Hei sinä siinä!",
                "Moi! Tervetuloa!",
                "Aa, hei!",
            ]
        else:
            phrases = [
                "Näen jonkun tuolla!",
                "Siellä on ihminen!",
                "Oho, joku siellä!",
            ]
        self.speak(random.choice(phrases))

    def announce_task(self, task: str):
        """Announce starting a task."""
        # Generate Finnish version of task
        context = {
            "action": f"Starting task: {task}",
            "mood": "eager",
        }
        comment = self.generator.generate(context, CommentaryType.NARRATION)
        if comment:
            self.speak(comment)
        else:
            self.speak(f"Okei, teen: {task[:30]}")

    def announce_done(self, result: str = ""):
        """Announce task completion."""
        phrases = [
            "Valmista tuli!",
            "Tehty!",
            "Siinä se on!",
            "Löysin!",
            "Homma hoidettu!",
        ]
        self.speak(random.choice(phrases))

    def comment_on_action(self, action_type: str, note: str = ""):
        """Comment on a specific action."""
        action_phrases = {
            "forward": ["Menen eteenpäin.", "Suoraan eteenpäin!"],
            "backward": ["Peruutan vähän.", "Taakse..."],
            "turn_left": ["Käännyn vasemmalle.", "Vasemmalle..."],
            "turn_right": ["Käännyn oikealle.", "Oikealle..."],
            "look_up": ["Katson ylös.", "Mitäs siellä ylhäällä?"],
            "look_down": ["Katson alas.", "Mitäs tuolla?"],
            "stop": ["Pysähdyn.", "Hetkinen..."],
            "approach": ["Menen lähemmäs...", "Tulen sinne päin."],
            "investigate": ["Tutkitaanpa...", "Mielenkiintoista!"],
        }

        phrases = action_phrases.get(action_type, FINNISH_PHRASES["thinking"])
        self.speak(random.choice(phrases))


# ========================
# Integration Helper
# ========================

def create_action_commentary(action_dict: Dict[str, Any], speaker: RobotSpeaker) -> Optional[str]:
    """
    Generate commentary for a robot action.

    Args:
        action_dict: The action being executed (drive, head, etc.)
        speaker: RobotSpeaker instance

    Returns:
        Commentary string if generated
    """
    action_type = action_dict.get("action", "")
    note = action_dict.get("note", "")

    # Determine action category
    if action_type == "drive":
        l = action_dict.get("l", 0)
        r = action_dict.get("r", 0)

        if l > 0 and r > 0:
            return speaker.maybe_comment(action="forward", mood="eager")
        elif l < 0 and r < 0:
            return speaker.maybe_comment(action="backward", mood="careful")
        elif l > 0 and r < 0:
            return speaker.maybe_comment(action="turn_right", mood="curious")
        elif l < 0 and r > 0:
            return speaker.maybe_comment(action="turn_left", mood="curious")

    elif action_type == "head":
        pos = action_dict.get("pos", 90)
        if pos < 80:
            return speaker.maybe_comment(action="look_up", mood="curious")
        elif pos > 100:
            return speaker.maybe_comment(action="look_down", mood="curious")

    elif action_type == "stop":
        return speaker.maybe_comment(action="stop", mood="thinking")

    elif action_type == "done":
        speaker.announce_done(action_dict.get("result", ""))
        return "Valmis!"

    return None


# ========================
# Test
# ========================

if __name__ == "__main__":
    print("Testing Robot Speech System...")

    speaker = RobotSpeaker()
    speaker.personality.chattiness = 1.0  # Always speak for testing
    speaker.personality.use_llm = True

    # Test greeting
    speaker.greet()
    time.sleep(2)

    # Test action comments
    test_contexts = [
        {"action": "exploring", "mood": "curious"},
        {"observation": "person detected", "mood": "social"},
        {"action": "turning", "mood": "curious"},
        {"observation": "doorway found", "mood": "eager"},
    ]

    for ctx in test_contexts:
        comment = speaker.maybe_comment(
            action=ctx.get("action", ""),
            observation=ctx.get("observation", ""),
            mood=ctx.get("mood", "neutral"),
            force=True
        )
        print(f"Context: {ctx} -> {comment}")
        time.sleep(2)

    print("\nDone!")
