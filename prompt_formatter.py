import logging
import re

logger = logging.getLogger(__name__)


class PromptFormatter:

    @staticmethod
    def format_for_generation(prompt: str) -> str:
        original = prompt.strip()
        formatted = original

        # What is/are patterns
        formatted = re.sub(r'^What is (.+?)\??\s*$', r'\1 is', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'^What are (.+?)\??\s*$', r'\1 are', formatted, flags=re.IGNORECASE)

        # Explain pattern
        formatted = re.sub(r'^Explain (.+?)\??\s*$', r'\1 is a concept that', formatted, flags=re.IGNORECASE)

        # Tell me about pattern
        formatted = re.sub(r'^Tell me about (.+?)\??\s*$', r'\1 refers to', formatted, flags=re.IGNORECASE)

        # How does pattern
        formatted = re.sub(r'^How does (.+?) work\??\s*$', r'\1 works by', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'^How do (.+?) work\??\s*$', r'\1 work by', formatted, flags=re.IGNORECASE)

        # Why is/are patterns
        formatted = re.sub(r'^Why is (.+?)\??\s*$', r'\1 is important because', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'^Why are (.+?)\??\s*$', r'\1 are important because', formatted, flags=re.IGNORECASE)

        # Describe pattern
        formatted = re.sub(r'^Describe (.+?)\??\s*$', r'\1 is', formatted, flags=re.IGNORECASE)

        # Define pattern
        formatted = re.sub(r'^Define (.+?)\??\s*$', r'\1 can be defined as', formatted, flags=re.IGNORECASE)

        # Compare pattern
        formatted = re.sub(r'^Compare (.+?) and (.+?)\??\s*$', r'\1 and \2 differ in that', formatted, flags=re.IGNORECASE)

        # List pattern
        formatted = re.sub(r'^List (.+?)\??\s*$', r'\1 include', formatted, flags=re.IGNORECASE)

        # Who is/was patterns
        formatted = re.sub(r'^Who is (.+?)\??\s*$', r'\1 is', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'^Who was (.+?)\??\s*$', r'\1 was', formatted, flags=re.IGNORECASE)

        # When did pattern
        formatted = re.sub(r'^When did (.+?)\??\s*$', r'\1 occurred', formatted, flags=re.IGNORECASE)

        # Where is/are patterns
        formatted = re.sub(r'^Where is (.+?)\??\s*$', r'\1 is located', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'^Where are (.+?)\??\s*$', r'\1 are located', formatted, flags=re.IGNORECASE)

        # Which pattern
        formatted = re.sub(r'^Which (.+?)\??\s*$', r'The \1 that', formatted, flags=re.IGNORECASE)

        # Can/Could patterns
        formatted = re.sub(r'^Can (.+?)\??\s*$', r'\1 can', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'^Could (.+?)\??\s*$', r'\1 could', formatted, flags=re.IGNORECASE)

        if formatted != original:
            logger.debug(f"Prompt transformation: '{original}' -> '{formatted}'")

        return formatted

    @staticmethod
    def should_log_transformation(original: str, formatted: str) -> bool:
        return original.strip() != formatted.strip()
