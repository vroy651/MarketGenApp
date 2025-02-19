
import streamlit as st
import json
import re
from datetime import datetime
from pathlib import Path

def validate_date_range(date_range_str):
    """Validates a date range string in "YYYY-MM-DD to YYYY-MM-DD" format."""
    try:
        if not date_range_str:  # Handle empty input
            return True  # Or False, depending on if you *require* a date range

        start_date_str, end_date_str = date_range_str.split(" to ")
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        if end_date < start_date:
            return False  # End date is before start date

        return True

    except ValueError:
        return False  # Invalid date format
    except Exception as e:
        print(f"Unexpected error in validate_date_range: {e}")  # Log for debugging
        return False

def validate_url(url):
    """Basic URL validation using a regular expression."""
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def validate_inputs(input_vars):
    """Validates user inputs from a dictionary.  Fields are optional."""

    errors = []
    campaign_name = input_vars.get("campaign_name")
    brand = input_vars.get("brand")
    sku = input_vars.get("sku")
    product_category = input_vars.get("product_category")
    tone_style = input_vars.get("tone_style")
    output_format = input_vars.get("output_format")
    promotion_link = input_vars.get("promotion_link")


    if not campaign_name:
        errors.append("Campaign Name is required.")
    if not brand:
        errors.append("Brand is required.")
    # ... (other required field checks) ...
    if not sku:
        errors.append("SKU is required.")
    if not product_category:
        errors.append("Product Category is required.")
    if not tone_style:
        errors.append("Tone & Style is required.")
    if not output_format:
        errors.append("Output Format is required.")

    if promotion_link and not validate_url(promotion_link):
        errors.append("Promotion Link must be a valid URL.")

    if errors:
        return False, "\n".join(errors)
    else:
        return True, ""
def save_content_to_file(content, campaign_name, save_format):
    """Saves the generated content to a file."""

    # Create the 'saved_campaigns' directory if it doesn't exist
    save_dir = Path("saved_campaigns")
    save_dir.mkdir(exist_ok=True)  #  Creates the directory if it doesn't exist

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp
    filename = save_dir / f"{campaign_name}_{timestamp}.{save_format}" # Use pathlib

    try:
        if save_format == "txt":
            with open(filename, "w") as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    f.write(str(content))
        elif save_format == "json":
            with open(filename, "w") as f:
                if isinstance(content, str):
                    json.dump({"content": content}, f, indent=4)
                else:
                    json.dump(content.to_dict(), f, indent=4)
        return str(filename) # Return string representation of path
    except Exception as e:
        st.error(f"Error saving content: {e}")
        return None

def load_campaign_template(template_name):
    """Loads a campaign template from a JSON file."""
    template_path = Path("templates") / f"{template_name}.json"

    try:
        with open(template_path, "r") as f:
            template_data = json.load(f)

            # --- Basic Template Validation (Example) ---
            required_keys = ["campaign_name", "sku", "product_category"] # Add more as needed
            for key in required_keys:
                if key not in template_data:
                    st.error(f"Template '{template_name}' is missing the required key: '{key}'")
                    return {}  # Return empty dict on error
            return template_data

    except FileNotFoundError:
        st.warning(f"Template '{template_name}' not found.  Using default values.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON in template '{template_name}'.")
        return {}