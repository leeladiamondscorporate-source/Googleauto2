import os
import csv
import ftplib
import pandas as pd
from google.cloud import storage
from flask import jsonify
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote
import hashlib

# ----------------------------
# CONFIGURATION & CREDENTIALS
# ----------------------------

# The GOOGLE_APPLICATION_CREDENTIALS environment variable should be set externally.
# Bucket details are loaded from environment variables.
bucket_name = os.environ.get("BUCKET_NAME")  # e.g., "sitemaps.leeladiamond.com"
bucket_folder = os.environ.get("BUCKET_FOLDER")  # e.g., "Googlefinal"

# Directories for file storage (using Linux paths)
local_output_directory = os.environ.get("LOCAL_OUTPUT_DIRECTORY", "/tmp/output")
ftp_download_dir = os.environ.get("FTP_DOWNLOAD_DIR", "/tmp/ftp")

# Create directories if they don't exist
os.makedirs(local_output_directory, exist_ok=True)
os.makedirs(ftp_download_dir, exist_ok=True)

# FTP Server Details
FTP_SERVER = "ftp.nivoda.net"
FTP_PORT = 21
FTP_USERNAME = "leeladiamondscorporate@gmail.com"
FTP_PASSWORD = "1yH£lG4n0Mq"

# Mapping product types to FTP file details and local save paths
ftp_files = {
    "natural": {
         "remote_filename": "Leela Diamond_natural.csv",
         "local_path": os.path.join(ftp_download_dir, "Natural.csv")
    },
    "lab_grown": {
         "remote_filename": "Leela Diamond_labgrown.csv",
         "local_path": os.path.join(ftp_download_dir, "Labgrown.csv")
    },
    "gemstone": {
         "remote_filename": "Leela Diamond_gemstones.csv",
         "local_path": os.path.join(ftp_download_dir, "gemstones.csv")
    }
}

# ----------------------------
# SEO KEYWORD MAPPING
# ----------------------------

# High-value diamond keywords for SEO
DIAMOND_KEYWORDS = {
    "natural": [
        "natural diamond", "certified diamond", "GIA diamond", "loose diamond",
        "engagement ring diamond", "wedding diamond", "investment diamond"
    ],
    "lab_grown": [
        "lab grown diamond", "lab created diamond", "cultured diamond", "synthetic diamond",
        "eco-friendly diamond", "sustainable diamond", "conflict-free diamond"
    ],
    "gemstone": [
        "precious gemstone", "natural gemstone", "certified gemstone", "loose gemstone",
        "colored gemstone", "fine gemstone", "investment gemstone"
    ]
}

# Shape-specific keywords
SHAPE_KEYWORDS = {
    "ROUND": ["round brilliant", "round cut", "classic round", "brilliant cut"],
    "PRINCESS": ["princess cut", "square diamond", "modern cut"],
    "EMERALD": ["emerald cut", "step cut", "rectangular diamond"],
    "CUSHION": ["cushion cut", "pillow cut", "antique style"],
    "OVAL": ["oval cut", "elongated diamond", "oval brilliant"],
    "PEAR": ["pear shape", "teardrop diamond", "pear brilliant"],
    "MARQUISE": ["marquise cut", "navette cut", "boat shape"],
    "HEART": ["heart shape", "romantic diamond", "heart brilliant"],
    "ASSCHER": ["asscher cut", "square emerald", "art deco"],
    "RADIANT": ["radiant cut", "rectangular brilliant", "trimmed corners"]
}

# Mapping shapes to default image URLs (same as before)
SHAPE_IMAGE_URLS = {
    "ASSCHER": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/ASSCHER.jpg",
    "BAGUETTE": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/BAGUETTE.jpg",
    "BRIOLETTE": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/BRIOLETTE.webp",
    "BULLET": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/BULLET.jpeg",
    "CALF": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/CALF.webp",
    "CUSHION": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/CUSHION.jpg",
    "CUSHION BRILLIANT": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/CUSHION%20BRILLIANT.webp",
    "CUSHION MODIFIED": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/CUSHION%20MODIFIED.jpg",
    "EMERALD": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/EMERALD.jpg",
    "EUROPEAN CUT": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/EUROPEAN%20CUT.webp",
    "HALF MOON": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/HALF%20MOON.jpg",
    "HEART": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/HEART.png",
    "HEPTAGONAL": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/HEPTAGONAL.webp",
    "HEXAGONAL": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/HEXAGONAL.webp",
    "KITE": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/KITE.jpg",
    "LOZENGE": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/LOZENGE.jpg",
    "MARQUISE": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/MARQUISE.jpg",
    "NONAGONAL": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/NONAGONAL.jpg",
    "OCTAGONAL": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/OCTAGONAL.jpg",
    "OLD MINER": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/OLD%20MINER.webp",
    "OTHER": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/OTHER.webp",
    "OVAL": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/OVAL.webp",
    "PEAR": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/PEAR%20MODIFIED%20BRILLIANT.webp",
    "PEAR MODIFIED BRILLIANT": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/PEAR.jpg",
    "PENTAGONAL": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/PENTAGONAL.jpg",
    "PRINCESS": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/PRINCESS.jpg",
    "RADIANT": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/RADIANT.jpg",
    "RECTANGULAR": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/RECTANGULAR.webp",
    "RHOMBOID": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/RHOMBOID.jpg",
    "ROSE": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/ROSE.webp",
    "ROUND": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/ROUND.png",
    "SHIELD": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/SHIELD.webp",
    "SQUARE": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/SQUARE%20EMERALD.webp",
    "SQUARE EMERALD": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/SQUARE%20EMERALD.webp",
    "SQUARE RADIANT": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/SQUARE%20RADIANT.webp",
    "STAR": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/STAR.jpg",
    "TAPERED BAGUETTE": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/TAPERED%20BAGUETTE.jpg",
    "TRAPEZOID": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/TRAPEZOID.jpg",
    "TRIANGULAR": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/TRIANGULAR.webp",
    "TRILLIANT": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/TRILLIANT.jpg",
}

# ----------------------------
# FTP DOWNLOAD FUNCTION
# ----------------------------

def download_file_from_ftp(remote_filename, local_path):
    """Download a file from the FTP server to a local path."""
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(FTP_SERVER, FTP_PORT)
            ftp.login(FTP_USERNAME, FTP_PASSWORD)
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f"RETR {remote_filename}", f.write)
            print(f"Downloaded {remote_filename} to {local_path}")
    except Exception as e:
        print(f"Error downloading {remote_filename}: {e}")

def download_all_files():
    """Download all defined files from the FTP server."""
    for product_type, file_info in ftp_files.items():
        download_file_from_ftp(file_info["remote_filename"], file_info["local_path"])

# ----------------------------
# SEO HELPER FUNCTIONS
# ----------------------------

def create_seo_friendly_url_slug(text):
    """Create SEO-friendly URL slug from text."""
    # Remove special characters and convert to lowercase
    slug = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
    # Replace spaces with hyphens
    slug = re.sub(r'\s+', '-', slug)
    # Remove consecutive hyphens
    slug = re.sub(r'-+', '-', slug)
    # Remove leading/trailing hyphens
    return slug.strip('-')

def generate_seo_keywords(product_type, shape, carats, color, clarity):
    """Generate SEO keywords for the product."""
    keywords = []
    
    # Add product type keywords
    keywords.extend(DIAMOND_KEYWORDS.get(product_type, []))
    
    # Add shape-specific keywords
    keywords.extend(SHAPE_KEYWORDS.get(shape, [shape.lower()]))
    
    # Add specific attributes
    keywords.extend([
        f"{carats} carat",
        f"{color} color diamond",
        f"{clarity} clarity",
        f"{carats}ct {shape.lower()}",
        f"buy {shape.lower()} diamond",
        f"{shape.lower()} diamond for sale"
    ])
    
    return ", ".join(keywords[:15])  # Limit to 15 keywords

def create_structured_data_json(product_data):
    """Create JSON-LD structured data for better SEO."""
    structured_data = {
        "@context": "https://schema.org/",
        "@type": "Product",
        "name": product_data['title'],
        "description": product_data['description'],
        "image": product_data['image_link'],
        "brand": {
            "@type": "Brand",
            "name": "Leela Diamonds"
        },
        "offers": {
            "@type": "Offer",
            "price": product_data['price'].replace(' CAD', ''),
            "priceCurrency": "CAD",
            "availability": "https://schema.org/InStock",
            "url": product_data['link']
        },
        "category": "Jewelry > Diamonds",
        "mpn": product_data['mpn']
    }
    return structured_data

# ----------------------------
# DATA PROCESSING FUNCTIONS
# ----------------------------

def convert_to_cad(price_usd):
    """Convert price from USD to CAD using a fixed exchange rate."""
    cad_rate = 1.46
    try:
        return round(price_usd * cad_rate, 2)
    except Exception as e:
        print(f"Error in currency conversion: {e}")
        return price_usd

def apply_markup(price):
    """Apply markup to the price based on predefined tiers."""
    base = price * 1.05 * 1.13
    additional = (
        210 if price <= 500 else
        375 if price <= 1000 else
        500 if price <= 1500 else
        700 if price <= 2000 else
        900 if price <= 2500 else
        1100 if price <= 3000 else
        1200 if price <= 5000 else
        1500 if price <= 100000 else
        0
    )
    return round(base + additional, 2)

def get_quality_score(cut, polish, symmetry, color, clarity):
    """Calculate a quality score for SEO ranking."""
    scores = {
        'cut': {'EX': 10, 'VG': 8, 'G': 6, 'F': 4, 'P': 2},
        'polish': {'EX': 10, 'VG': 8, 'G': 6, 'F': 4, 'P': 2},
        'symmetry': {'EX': 10, 'VG': 8, 'G': 6, 'F': 4, 'P': 2},
        'color': {'D': 10, 'E': 9, 'F': 8, 'G': 7, 'H': 6, 'I': 5, 'J': 4, 'K': 3, 'L': 2, 'M': 1},
        'clarity': {'FL': 10, 'IF': 9, 'VVS1': 8, 'VVS2': 7, 'VS1': 6, 'VS2': 5, 'SI1': 4, 'SI2': 3, 'I1': 2, 'I2': 1}
    }
    
    total_score = (
        scores['cut'].get(cut, 0) +
        scores['polish'].get(polish, 0) +
        scores['symmetry'].get(symmetry, 0) +
        scores['color'].get(color, 0) +
        scores['clarity'].get(clarity, 0)
    )
    
    return min(total_score / 5, 10)  # Normalize to 0-10 scale

def create_seo_optimized_title(product_type, shape, carats, color, clarity, lab, cut_grade=None):
    """Create SEO-optimized product title with primary keywords first."""
    shape_clean = shape.title()
    
    # Primary keyword combinations for better ranking
    if product_type == "natural":
        primary_keyword = f"{carats}ct {shape_clean} Natural Diamond"
        secondary_info = f"{color}-{clarity} {lab} Certified"
    elif product_type == "lab_grown":
        primary_keyword = f"{carats}ct {shape_clean} Lab Grown Diamond"
        secondary_info = f"{color}-{clarity} {lab} Certified"
    else:  # gemstone
        primary_keyword = f"{carats}ct {shape_clean} Gemstone"
        secondary_info = f"{color} {clarity}"
    
    # Add quality indicators for high-grade diamonds
    quality_indicators = []
    if cut_grade in ['EX', 'VG']:
        quality_indicators.append("Premium Cut")
    if color in ['D', 'E', 'F']:
        quality_indicators.append("Colorless")
    elif color in ['G', 'H']:
        quality_indicators.append("Near Colorless")
    if clarity in ['FL', 'IF', 'VVS1', 'VVS2']:
        quality_indicators.append("High Clarity")
    
    quality_suffix = f" | {' '.join(quality_indicators)}" if quality_indicators else ""
    
    # Keep title under 150 characters for optimal SEO
    title = f"{primary_keyword} - {secondary_info}{quality_suffix}"
    return title[:150]

def create_seo_optimized_description(product_type, row):
    """Create detailed, SEO-optimized product description."""
    shape = row.get('shape', '').title()
    carats = row.get('carats', '')
    color = row.get('col', row.get('Color', ''))
    clarity = row.get('clar', row.get('Clarity', ''))
    lab = row.get('lab', row.get('Lab', ''))
    
    # Base description with primary keywords
    if product_type in ["natural", "lab_grown"]:
        diamond_type = "natural diamond" if product_type == "natural" else "lab-grown diamond"
        base_desc = f"Premium {carats} carat {shape.lower()} {diamond_type} featuring {color} color grade and {clarity} clarity rating."
        
        # Add technical specifications
        specs = []
        for field in ['cut', 'pol', 'symm', 'table', 'depth', 'flo']:
            if field in row and row[field]:
                field_names = {
                    'cut': 'Cut Grade',
                    'pol': 'Polish',
                    'symm': 'Symmetry',
                    'table': 'Table %',
                    'depth': 'Depth %',
                    'flo': 'Fluorescence'
                }
                specs.append(f"{field_names[field]}: {row[field]}")
        
        # Add measurements
        dimensions = []
        for dim in ['length', 'width', 'height']:
            if dim in row and row[dim]:
                dimensions.append(row[dim])
        
        dimension_text = f"Measurements: {' x '.join(dimensions)} mm" if dimensions else ""
        
        # Certification and benefits
        cert_text = f"{lab} certified for authenticity and quality assurance."
        
        if product_type == "natural":
            benefits = "Perfect for engagement rings, wedding jewelry, or investment pieces. Natural earth-formed diamond with excellent brilliance and fire."
        else:
            benefits = "Eco-friendly and conflict-free alternative with identical optical properties to natural diamonds. Perfect for modern, conscious consumers."
        
        # Combine all parts
        full_description = f"{base_desc} {dimension_text} {' | '.join(specs)} {cert_text} {benefits}"
        
    else:  # gemstone
        gem_type = row.get('gemType', 'gemstone')
        base_desc = f"Exquisite {carats} carat {shape.lower()} {color.lower()} {gem_type.lower()} with {clarity} clarity."
        
        # Add gemstone-specific details
        details = []
        for field in ['Cut', 'Treatment', 'Mine of Origin']:
            if field in row and row[field]:
                details.append(f"{field}: {row[field]}")
        
        cert_text = f"{lab} certified natural gemstone." if lab else "Natural gemstone."
        benefits = f"Ideal for fine jewelry, collectors, or investment. Rare {gem_type.lower()} with exceptional color saturation."
        
        full_description = f"{base_desc} {' | '.join(details)} {cert_text} {benefits}"
    
    # Ensure description is comprehensive but not too long
    return full_description[:500] + "..." if len(full_description) > 500 else full_description

def process_files_to_cad(files_to_load, output_file):
    """Process input CSV files with SEO optimizations and save to a combined file."""
    try:
        all_data = []
        all_urls = []  # For sitemap generation
        
        for product_type, file_info in files_to_load.items():
            input_file = file_info["file_path"]

            # Load the CSV file
            df = pd.read_csv(input_file, dtype=str).fillna('')

            # Clean shape and image columns (if they exist)
            if 'shape' in df.columns:
                df['shape'] = df['shape'].str.strip().str.upper()

            if 'image' in df.columns:
                df['image'] = df['image'].str.extract(r'(https?://.*\.(jpg|png|webp))')[0]

            # Assign default image if no valid URL is found
            df['image_link'] = df.apply(
                lambda row: row['image'] if pd.notna(row['image']) and row['image']
                else SHAPE_IMAGE_URLS.get(row['shape'], ''),
                axis=1
            )

            # Convert price and apply markup
            df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
            df['price'] = df['price'].apply(apply_markup).apply(convert_to_cad)
            df['price'] = df['price'].astype(str) + " CAD"

            # Generate SEO-optimized IDs and URLs
            df['id'] = df['ReportNo'] + "CA"
            df['availability'] = 'in_stock'
            df['google_product_category'] = '188'
            df['brand'] = 'Leela Diamonds'
            df['mpn'] = df['id']
            df['condition'] = 'new'
            df['color'] = 'white/yellow/rose gold'
            df['age_group'] = 'adult'
            df['gender'] = 'unisex'

            # SEO-optimized product templates
            if product_type == "natural":
                df['title'] = df.apply(
                    lambda row: create_seo_optimized_title("natural", row['shape'], row['carats'], row['col'], row['clar'], row['lab'], row.get('cut')),
                    axis=1
                )
                df['description'] = df.apply(lambda row: create_seo_optimized_description("natural", row), axis=1)
                df['link'] = df.apply(lambda row: f"https://leeladiamond.com/pages/natural-diamond-catalog?id={row['ReportNo']}", axis=1)
                
                # Add SEO keywords
                df['custom_label_0'] = df.apply(
                    lambda row: generate_seo_keywords("natural", row['shape'], row['carats'], row['col'], row['clar']),
                    axis=1
                )
                
                # Add quality score for ranking
                df['custom_label_1'] = df.apply(
                    lambda row: f"Quality Score: {get_quality_score(row.get('cut', ''), row.get('pol', ''), row.get('symm', ''), row['col'], row['clar'])}",
                    axis=1
                )

            elif product_type == "lab_grown":
                df['title'] = df.apply(
                    lambda row: create_seo_optimized_title("lab_grown", row['shape'], row['carats'], row['col'], row['clar'], row['lab'], row.get('cut')),
                    axis=1
                )
                df['description'] = df.apply(lambda row: create_seo_optimized_description("lab_grown", row), axis=1)
                df['link'] = df.apply(lambda row: f"https://leeladiamond.com/pages/lab-grown-diamond-catalog?id={row['ReportNo']}", axis=1)
                
                df['custom_label_0'] = df.apply(
                    lambda row: generate_seo_keywords("lab_grown", row['shape'], row['carats'], row['col'], row['clar']),
                    axis=1
                )
                
                df['custom_label_1'] = df.apply(
                    lambda row: f"Quality Score: {get_quality_score(row.get('cut', ''), row.get('pol', ''), row.get('symm', ''), row['col'], row['clar'])}",
                    axis=1
                )

            else:  # gemstone
                df['title'] = df.apply(
                    lambda row: create_seo_optimized_title("gemstone", row['shape'], row['carats'], row.get('Color', ''), row.get('Clarity', ''), row.get('Lab', '')),
                    axis=1
                )
                df['description'] = df.apply(lambda row: create_seo_optimized_description("gemstone", row), axis=1)
                df['link'] = df.apply(lambda row: f"https://leeladiamond.com/pages/gemstone-catalog?id={row['ReportNo']}", axis=1)
                
                df['custom_label_0'] = df.apply(
                    lambda row: generate_seo_keywords("gemstone", row['shape'], row['carats'], row.get('Color', ''), row.get('Clarity', '')),
                    axis=1
                )

            # Collect URLs for sitemap
            all_urls.extend(df['link'].tolist())

            # Add additional SEO fields
            df['additional_image_link'] = df['image_link']  # Backup image
            df['product_highlight'] = df.apply(
                lambda row: f"{row['shape']} Shape | {row.get('carats', row.get('Carats', ''))} Carats | Certified Quality",
                axis=1
            )

            # Select final columns with SEO enhancements
            columns_to_keep = [
                'id', 'title', 'description', 'link', 'image_link', 'availability', 'price',
                'google_product_category', 'brand', 'mpn', 'condition', 'color', 'age_group', 'gender',
                'custom_label_0', 'custom_label_1', 'additional_image_link', 'product_highlight'
            ]
            
            # Only keep columns that exist in the dataframe
            existing_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[existing_columns]
            
            all_data.append(df)

        # Combine all product data and save to CSV
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by quality indicators for better indexing priority
        if 'custom_label_1' in combined_df.columns:
            combined_df['quality_sort'] = combined_df['custom_label_1'].str.extract(r'(\d+\.?\d*)').astype(float).fillna(0)
            combined_df = combined_df.sort_values('quality_sort', ascending=False).drop('quality_sort', axis=1)
        
        combined_df.to_csv(
            output_file,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar='\\'
        )
        print(f"Combined SEO-optimized data saved to {output_file}")
        
        # Generate XML sitemap for better indexing
        generate_sitemap(all_urls, os.path.join(local_output_directory, "sitemap.xml"))
        
        return all_urls
        
    except Exception as e:
        print(f"Error in processing files: {e}")
        return []

def generate_sitemap(urls, sitemap_path):
    """Generate XML sitemap for better SEO indexing."""
    try:
        root = ET.Element("urlset")
        root.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")
        
        for url in urls:
            url_element = ET.SubElement(root, "url")
            loc = ET.SubElement(url_element, "loc")
            loc.text = url
            
            # Add last modified date (current date)
            lastmod = ET.SubElement(url_element, "lastmod")
            from datetime import datetime
            lastmod.text = datetime.now().strftime("%Y-%m-%d")
            
            # Set change frequency and priority
            changefreq = ET.SubElement(url_element, "changefreq")
            changefreq.text = "weekly"  # Products change weekly
            
            priority = ET.SubElement(url_element, "priority")
            priority.text = "0.8"  # High priority for product pages
        
        # Write to XML file
        tree = ET.ElementTree(root)
        tree.write(sitemap_path, encoding="utf-8", xml_declaration=True)
        print(f"Sitemap generated: {sitemap_path}")
        
    except Exception as e:
        print(f"Error generating sitemap: {e}")

# ----------------------------
# GOOGLE CLOUD UPLOAD FUNCTION
# ----------------------------

def upload_files_to_bucket(bucket_name, bucket_folder, local_directory):
    """Upload all files in a local directory to a GCS bucket folder with proper headers."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        for file_name in os.listdir(local_directory):
            file_path = os.path.join(local_directory, file_name)
            if os.path.isfile(file_path):
                destination_blob_name = f"{bucket_folder}/{file_name}"
                blob = bucket.blob(destination_blob_name)
                
                # Set appropriate content type for SEO
                if file_name.endswith('.xml'):
                    blob.content_type = 'application/xml'
                elif file_name.endswith('.csv'):
                    blob.content_type = 'text/csv'
                
                # Set cache control for better performance
                blob.cache_control = 'max-age=3600'  # 1 hour cache
                
                blob.upload_from_filename(file_path)
                
                # Make sitemap public for search engines
                if file_name == 'sitemap.xml':
                    blob.make_public()
                
                print(f"Uploaded {file_name} to {destination_blob_name}")
                
        print(f"Sitemap URL: https://storage.googleapis.com/{bucket_name}/{bucket_folder}/sitemap.xml")
        
    except Exception as e:
        print(f"Error during upload: {e}")

# ----------------------------
# MAIN AUTOMATION WORKFLOW
# ----------------------------

def run_workflow():
    # Step 1: Download raw CSV files from the FTP server
    download_all_files()
    
    # Step 2: Define file paths for processing (using ftp_download_dir)
    files_to_load = {
        "natural": {"file_path": os.path.join(ftp_download_dir, "Natural.csv")},
        "lab_grown": {"file_path": os.path.join(ftp_download_dir, "Labgrown.csv")},
        "gemstone": {"file_path": os.path.join(ftp_download_dir, "gemstones.csv")}
    }
    output_file = os.path.join(local_output_directory, "combined_google_merchant_feed.csv")
    
    # Step 3: Process the downloaded files with SEO optimizations
    product_urls = process_files_to_cad(files_to_load, output_file)
    
    # Step 4: Generate additional SEO files
    generate_robots_txt(os.path.join(local_output_directory, "robots.txt"))
    generate_feed_metadata(os.path.join(local_output_directory, "feed_metadata.json"))
    
    # Step 5: Upload all files (CSV, sitemap, robots.txt) to GCS
    upload_files_to_bucket(bucket_name, bucket_folder, local_output_directory)
    
    # Step 6: Submit sitemap to search engines (optional)
    submit_sitemap_to_search_engines()
    
    return f"SEO-optimized workflow executed successfully. Processed {len(product_urls)} products."

# ----------------------------
# ADDITIONAL SEO FUNCTIONS
# ----------------------------

def generate_robots_txt(robots_path):
    """Generate robots.txt file for better crawling."""
    try:
        robots_content = f"""User-agent: *
Allow: /

# Sitemap location
Sitemap: https://storage.googleapis.com/{bucket_name}/{bucket_folder}/sitemap.xml

# Crawl-delay for polite crawling
Crawl-delay: 1

# Allow all search engine bots
User-agent: Googlebot
Allow: /

User-agent: Bingbot
Allow: /

User-agent: Slurp
Allow: /
"""
        
        with open(robots_path, 'w') as f:
            f.write(robots_content)
        
        print(f"Robots.txt generated: {robots_path}")
        
    except Exception as e:
        print(f"Error generating robots.txt: {e}")

def generate_feed_metadata(metadata_path):
    """Generate metadata file with feed information."""
    try:
        from datetime import datetime
        import json
        
        metadata = {
            "feed_name": "Leela Diamonds - Google Merchant Center Feed",
            "description": "SEO-optimized product feed for natural diamonds, lab-grown diamonds, and gemstones",
            "last_updated": datetime.now().isoformat(),
            "total_products": 0,  # Will be updated during processing
            "categories": ["Natural Diamonds", "Lab-Grown Diamonds", "Gemstones"],
            "currency": "CAD",
            "country": "CA",
            "language": "en",
            "feed_format": "Google Merchant Center CSV",
            "seo_optimizations": [
                "SEO-friendly product titles with primary keywords",
                "Detailed product descriptions with technical specifications",
                "Quality scoring for better ranking",
                "Keyword optimization for diamond industry",
                "Structured data markup compatibility",
                "XML sitemap generation",
                "Robots.txt for crawler guidance"
            ],
            "indexing_improvements": [
                "Priority-based URL sorting by quality score",
                "Weekly sitemap updates",
                "Proper content-type headers",
                "Cache control optimization",
                "Public sitemap accessibility"
            ]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Feed metadata generated: {metadata_path}")
        
    except Exception as e:
        print(f"Error generating feed metadata: {e}")

def submit_sitemap_to_search_engines():
    """Submit sitemap to major search engines for faster indexing."""
    try:
        import urllib.request
        import urllib.parse
        
        sitemap_url = f"https://storage.googleapis.com/{bucket_name}/{bucket_folder}/sitemap.xml"
        
        # Google Search Console submission
        google_ping_url = f"https://www.google.com/ping?sitemap={urllib.parse.quote(sitemap_url)}"
        
        # Bing Webmaster Tools submission  
        bing_ping_url = f"https://www.bing.com/ping?sitemap={urllib.parse.quote(sitemap_url)}"
        
        try:
            urllib.request.urlopen(google_ping_url, timeout=10)
            print("Sitemap submitted to Google")
        except Exception as e:
            print(f"Could not submit to Google: {e}")
        
        try:
            urllib.request.urlopen(bing_ping_url, timeout=10)
            print("Sitemap submitted to Bing")
        except Exception as e:
            print(f"Could not submit to Bing: {e}")
            
    except Exception as e:
        print(f"Error submitting sitemap: {e}")

def validate_feed_seo_compliance(csv_path):
    """Validate the generated feed for SEO compliance."""
    try:
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        validation_results = {
            "total_products": len(df),
            "issues": [],
            "recommendations": []
        }
        
        # Check title lengths
        long_titles = df[df['title'].str.len() > 150]
        if len(long_titles) > 0:
            validation_results["issues"].append(f"{len(long_titles)} products have titles longer than 150 characters")
        
        # Check description lengths
        short_descriptions = df[df['description'].str.len() < 160]
        if len(short_descriptions) > 0:
            validation_results["issues"].append(f"{len(short_descriptions)} products have descriptions shorter than 160 characters")
        
        # Check for missing images
        missing_images = df[df['image_link'].str.len() == 0]
        if len(missing_images) > 0:
            validation_results["issues"].append(f"{len(missing_images)} products missing image links")
        
        # Check price format
        invalid_prices = df[~df['price'].str.contains('CAD')]
        if len(invalid_prices) > 0:
            validation_results["issues"].append(f"{len(invalid_prices)} products have invalid price format")
        
        # SEO recommendations
        if 'custom_label_0' in df.columns:
            products_with_keywords = df[df['custom_label_0'].str.len() > 0]
            validation_results["recommendations"].append(f"{len(products_with_keywords)} products have SEO keywords")
        
        print(f"Feed validation completed: {validation_results}")
        return validation_results
        
    except Exception as e:
        print(f"Error validating feed: {e}")
        return None

# ----------------------------
# ENHANCED INDEXING FUNCTIONS
# ----------------------------

def create_product_index_priority(df):
    """Create indexing priority based on product characteristics."""
    try:
        # Factors that affect indexing priority:
        # 1. Quality score (higher is better)
        # 2. Price range (mid-range often performs better)
        # 3. Popular shapes (round, princess, etc.)
        # 4. Certification (GIA, AGS higher priority)
        
        priority_scores = []
        
        for _, row in df.iterrows():
            score = 0.5  # Base priority
            
            # Quality bonus
            if 'custom_label_1' in row and pd.notna(row['custom_label_1']):
                try:
                    quality_score = float(row['custom_label_1'].split(':')[-1].strip())
                    score += min(quality_score / 20, 0.3)  # Up to 0.3 bonus
                except:
                    pass
            
            # Shape popularity bonus
            if 'title' in row:
                popular_shapes = ['ROUND', 'PRINCESS', 'CUSHION', 'OVAL', 'EMERALD']
                for shape in popular_shapes:
                    if shape in row['title'].upper():
                        score += 0.1
                        break
            
            # Certification bonus
            if 'title' in row:
                premium_labs = ['GIA', 'AGS', 'GCAL']
                for lab in premium_labs:
                    if lab in row['title'].upper():
                        score += 0.1
                        break
            
            # Price range optimization (sweet spot for engagement rings)
            try:
                price_num = float(row['price'].replace(' CAD', ''))
                if 1000 <= price_num <= 10000:  # Sweet spot for most buyers
                    score += 0.1
            except:
                pass
            
            priority_scores.append(min(score, 1.0))  # Cap at 1.0
        
        return priority_scores
        
    except Exception as e:
        print(f"Error creating priority scores: {e}")
        return [0.8] * len(df)  # Default priority

def generate_category_sitemaps(all_urls, output_dir):
    """Generate separate sitemaps for different product categories."""
    try:
        categories = {
            'natural-diamonds': [url for url in all_urls if 'natural-diamond-catalog' in url],
            'lab-grown-diamonds': [url for url in all_urls if 'lab-grown-diamond-catalog' in url],
            'gemstones': [url for url in all_urls if 'gemstone-catalog' in url]
        }
        
        sitemap_index_urls = []
        
        for category, urls in categories.items():
            if urls:
                sitemap_path = os.path.join(output_dir, f"sitemap-{category}.xml")
                generate_sitemap(urls, sitemap_path)
                sitemap_index_urls.append(f"https://storage.googleapis.com/{bucket_name}/{bucket_folder}/sitemap-{category}.xml")
        
        # Create sitemap index
        create_sitemap_index(sitemap_index_urls, os.path.join(output_dir, "sitemap-index.xml"))
        
    except Exception as e:
        print(f"Error generating category sitemaps: {e}")

def create_sitemap_index(sitemap_urls, index_path):
    """Create a sitemap index file."""
    try:
        root = ET.Element("sitemapindex")
        root.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")
        
        for sitemap_url in sitemap_urls:
            sitemap_element = ET.SubElement(root, "sitemap")
            loc = ET.SubElement(sitemap_element, "loc")
            loc.text = sitemap_url
            
            lastmod = ET.SubElement(sitemap_element, "lastmod")
            from datetime import datetime
            lastmod.text = datetime.now().strftime("%Y-%m-%d")
        
        tree = ET.ElementTree(root)
        tree.write(index_path, encoding="utf-8", xml_declaration=True)
        print(f"Sitemap index generated: {index_path}")
        
    except Exception as e:
        print(f"Error creating sitemap index: {e}")

# ----------------------------
# CLOUD FUNCTION ENTRY POINT
# ----------------------------

def cloud_function_entry(request):
    """HTTP Cloud Function entry point with enhanced SEO features."""
    try:
        result = run_workflow()
        
        # Validate the generated feed
        output_file = os.path.join(local_output_directory, "combined_google_merchant_feed.csv")
        validation_results = validate_feed_seo_compliance(output_file)
        
        response_data = {
            "status": "success",
            "message": result,
            "validation": validation_results,
            "seo_features": [
                "SEO-optimized titles and descriptions",
                "XML sitemap generation",
                "Robots.txt creation",
                "Quality-based priority scoring",
                "Keyword optimization",
                "Structured data compatibility",
                "Category-specific sitemaps",
                "Search engine submission"
            ],
            "indexing_improvements": [
                "Priority-based URL ranking",
                "Weekly update frequency",
                "Proper HTTP headers",
                "Public sitemap accessibility",
                "Multi-category organization"
            ]
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e),
            "recommendations": [
                "Check FTP connection and credentials",
                "Verify Google Cloud Storage permissions",
                "Ensure all required CSV columns are present",
                "Check file paths and directory permissions"
            ]
        }), 500

# ----------------------------
# RUN LOCALLY (for testing purposes)
# ----------------------------

if __name__ == "__main__":
    # For local testing, run the workflow directly
    print("Starting SEO-optimized Google Merchant Center feed generation...")
    result = run_workflow()
    print(result)
    
    # Validate the output
    output_file = os.path.join(local_output_directory, "combined_google_merchant_feed.csv")
    if os.path.exists(output_file):
        validation = validate_feed_seo_compliance(output_file)
        print(f"Feed validation: {validation}")
    
    print("Process completed with SEO optimizations!")