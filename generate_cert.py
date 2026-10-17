import ipaddress
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
import sys

def generate_self_signed_cert(ip_address="192.168.0.138"):
    print(f"Generating self-signed certificate for {ip_address}...")
    
    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Set subject and issuer
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, u"Semantic JSCC Local"),
    ])
    
    # Build certificate
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([x509.IPAddress(ipaddress.IPv4Address(ip_address))]),
        critical=False,
    ).sign(key, hashes.SHA256())
    
    # Write private key
    with open("key.pem", "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        
    # Write certificate
    with open("cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
        
    print("Successfully generated key.pem and cert.pem!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_self_signed_cert(sys.argv[1])
    else:
        generate_self_signed_cert()
