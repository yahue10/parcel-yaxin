# here we find the constants of the project, in particular the volumes of the elements (letters, parcels, newspapers)

# Volumes of the elements (in L)
V_letter = 0.3 # 1 letter (14x9x1) minimal
V_newspaper = 1.5 # (41x29x1)+...+.. : average format  (tabloid, berliner...)
V_ordinaire = 5.4 #  colis (30x10x18)

# links between the volumes and the names of the elements
PACTE = V_letter * 1  # ratio simplifié
UTE = V_ordinaire
SIROP = V_newspaper

dico_vol = {
    "PACTE": PACTE,
    "UTE": UTE,
    "SIROP": SIROP
}

# volumes bikes (in L)
V_VAE =  200 # electric bike --> 2xsac a dos
V_VCAE =  540 # cargo bike - 100 colis

# volumes of the vehicles (in L)
V_Berlingo = 1050
V_E_Jumpy = 3800
V_Jumpy = 5000
V_Kangoo = 3600
V_Kangoo_ZE = 3200
V_Master_ZE = 12000