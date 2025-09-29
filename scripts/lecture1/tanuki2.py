words = str(input())

def tanuki2(c):
    s = ""
    for c in words:
        if c != "た":
            s += c
    return s

print(tanuki2(words))