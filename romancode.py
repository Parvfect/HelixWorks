a = 67 ** (851)
b = 2**a.bit_length()

print(b)
#b = int.from_bytes(bytearray(file_handle.read()), byteorder='big')
quot = b
lst = []
while quot > 0:
    quot, rem = divmod(quot, 67)
    lst.append(rem)

print(lst)
print(len(lst))