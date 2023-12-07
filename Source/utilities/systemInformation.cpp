#include <bitset>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unistd.h>

#include "systemInformation.hpp"

namespace adaptive {

uint64_t l3cacheSize() { return sysconf(_SC_LEVEL3_CACHE_SIZE); }

uint32_t bytesPerCacheLine() { return sysconf(_SC_LEVEL1_DCACHE_LINESIZE); }

uint32_t logicalCoresCount() { return static_cast<uint32_t>(std::thread::hardware_concurrency()); }

std::string getProjectRootDirectory() {
  namespace fs = std::filesystem;
  return fs::current_path().parent_path().string();
}

void printTlbSpecificationsFromLeaf0x18(unsigned int maxSubleaf) {
  unsigned int leaf = 0x18;
  unsigned int eax, ebx, ecx, edx;

  for(unsigned int subleaf = 0x1; subleaf <= maxSubleaf; subleaf++) {
    eax = leaf;
    ecx = subleaf;

    asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));

    printf("0x%08x-", eax);
    printf("0x%08x-", ebx);
    printf("0x%08x-", ecx);
    printf("0x%08x", edx);
    std::cout << " [SL 0" << subleaf << "]" << std::endl;
  }
  std::cout << std::endl;

  for(unsigned int subleaf = 0x1; subleaf <= maxSubleaf; subleaf++) {
    eax = leaf;
    ecx = subleaf;

    asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));

    std::string eax_binary = std::bitset<32>(eax).to_string();
    std::string ebx_binary = std::bitset<32>(ebx).to_string();
    std::string ecx_binary = std::bitset<32>(ecx).to_string();
    std::string edx_binary = std::bitset<32>(edx).to_string();

    for(int i = 4; i < 39; i += 5) {
      eax_binary.insert(i, " ");
      ebx_binary.insert(i, " ");
      ecx_binary.insert(i, " ");
      edx_binary.insert(i, " ");
    }

    std::cout << eax_binary << "---";
    std::cout << ebx_binary << "---";
    std::cout << ecx_binary << "---";
    std::cout << edx_binary;
    std::cout << " [SL 0" << subleaf << "]" << std::endl;
  }
  std::cout << std::endl;

  for(unsigned int subleaf = 0x1; subleaf <= maxSubleaf; subleaf++) {
    eax = leaf;
    ecx = subleaf;

    asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));

    unsigned int translationCacheLevel = (edx >> 5) & 0x7;
    unsigned int translationCacheType = edx & 0x1F;
    unsigned int waysOfAssociativity = (ebx >> 16) & 0xFF;
    unsigned int numberOfSets = ecx;
    bool supports4KPage = ebx & 0x1;
    bool supports2MBPage = (ebx >> 1) & 0x1;
    bool supports4MBPage = (ebx >> 2) & 0x1;
    bool supports1GBPage = (ebx >> 3) & 0x1;

    std::cout << "L" << translationCacheLevel;
    if(translationCacheType == 1)
      std::cout << " Data TLB, ";
    if(translationCacheType == 2)
      std::cout << " Instruction TLB, ";
    if(translationCacheType == 3)
      std::cout << " Unified TLB, ";
    if(translationCacheType == 4)
      std::cout << " Load Only TLB, ";
    if(translationCacheType == 5)
      std::cout << " Store Only TLB, ";
    std::cout << waysOfAssociativity << " way associativity, ";
    std::cout << numberOfSets << " sets, ";
    std::cout << waysOfAssociativity * numberOfSets << " entries, ";
    std::cout << "supports ";
    if(supports4KPage)
      std::cout << "4K ";
    if(supports2MBPage)
      std::cout << "2MB ";
    if(supports4MBPage)
      std::cout << "4MB ";
    if(supports1GBPage)
      std::cout << "1GB ";
    std::cout << "page sizes" << std::endl;
  }
}

void printByteDescriptorFromLeaf0x2(unsigned char value) {
  switch(value) {
  case 0x00:
    std::cout << "Null descriptor";
    break;
  case 0x01:
    std::cout << "Instruction TLB: 4K, 4-way, 32 entries";
    break;
  case 0x02:
    std::cout << "Instruction TLB: 4M, fully associative, 2 entries";
    break;
  case 0x03:
    std::cout << "Data TLB: 4K, 4-way, 64 entries";
    break;
  case 0x04:
    std::cout << "Data TLB: 4M, 4-way, 8 entries";
    break;
  case 0x05:
    std::cout << "Data TLB1: 4M, 4-way, 32 entries";
    break;
  case 0x06:
    std::cout << "1st-level instruction cache: 8KB, 4-way, 32 byte line size";
    break;
  case 0x08:
    std::cout << "1st-level instruction cache: 16KB, 4-way, 32 byte line size";
    break;
  case 0x09:
    std::cout << "1st-level instruction cache: 32KB, 4-way, 64 byte line size";
    break;
  case 0x0A:
    std::cout << "1st-level data cache: 8KB, 2-way, 32 byte line size";
    break;
  case 0x0B:
    std::cout << "Instruction TLB: 4M, 4-way, 4 entries";
    break;
  case 0x0C:
    std::cout << "1st-level data cache: 16KB, 4-way, 32 byte line size";
    break;
  case 0x0D:
    std::cout << "1st-level data cache: 16KB, 4-way, 64 byte line size";
    break;
  case 0x0E:
    std::cout << "1st-level data cache: 24KB, 6-way, 64 byte line size";
    break;
  case 0x1D:
    std::cout << "2nd-level cache: 128KB, 2-way, 64 byte line size";
    break;
  case 0x21:
    std::cout << "2nd-level cache: 256KB, 8-way, 64 byte line size";
    break;
  case 0x22:
    std::cout << "3rd-level cache: 512KB, 4-way, 64 byte line size, 2 lines per sector";
    break;
  case 0x23:
    std::cout << "3rd-level cache: 1MB, 8-way, 64 byte line size, 2 lines per sector";
    break;
  case 0x24:
    std::cout << "2nd-level cache: 1MB, 16-way, 64 byte line size";
    break;
  case 0x25:
    std::cout << "3rd-level cache: 2MB, 8-way, 64 byte line size, 2 lines per sector";
    break;
  case 0x29:
    std::cout << "3rd-level cache: 4MB, 8-way, 64 byte line size, 2 lines per sector";
    break;
  case 0x2C:
    std::cout << "1st-level data cache: 32KB, 8-way, 64 byte line size";
    break;
  case 0x30:
    std::cout << "1st-level instruction cache: 32KB, 8-way, 64 byte line size";
    break;
  case 0x40:
    std::cout << "No 2nd-level cache or, if valid, no 3rd-level cache";
    break;
  case 0x41:
    std::cout << "2nd-level cache: 128KB, 4-way, 32 byte line size";
    break;
  case 0x42:
    std::cout << "2nd-level cache: 256KB, 4-way, 32 byte line size";
    break;
  case 0x43:
    std::cout << "2nd-level cache: 512KB, 4-way, 32 byte line size";
    break;
  case 0x44:
    std::cout << "2nd-level cache: 1MB, 4-way, 32 byte line size";
    break;
  case 0x45:
    std::cout << "2nd-level cache: 2MB, 4-way, 32 byte line size";
    break;
  case 0x46:
    std::cout << "3rd-level cache: 4MB, 4-way, 64 byte line size";
    break;
  case 0x47:
    std::cout << "3rd-level cache: 8MB, 8-way, 64 byte line size";
    break;
  case 0x48:
    std::cout << "2nd-level cache: 3MB, 12-way, 64 byte line size";
    break;
  case 0x49:
    std::cout << "3rd-level cache: 4MB, 16-way, 64 byte line size";
    break;
  case 0x4A:
    std::cout << "3rd-level cache: 6MB, 12-way, 64 byte line size";
    break;
  case 0x4B:
    std::cout << "3rd-level cache: 8MB, 16-way, 64 byte line size";
    break;
  case 0x4C:
    std::cout << "3rd-level cache: 12MB, 12-way, 64 byte line size";
    break;
  case 0x4D:
    std::cout << "3rd-level cache: 16MB, 16-way, 64 byte line size";
    break;
  case 0x4E:
    std::cout << "2nd-level cache: 6MB, 24-way, 64 byte line size";
    break;
  case 0x4F:
    std::cout << "Instruction TLB: 4K, 32 entries";
    break;
  case 0x50:
    std::cout << "Instruction TLB: 4K/2M pages, 64 entries";
    break;
  case 0x51:
    std::cout << "Instruction TLB: 4K/2M pages, 128 entries";
    break;
  case 0x52:
    std::cout << "Instruction TLB: 4K/2M pages, 256 entries";
    break;
  case 0x55:
    std::cout << "Instruction TLB: 2M/4M pages, fully associative, 7 entries";
    break;
  case 0x56:
    std::cout << "Data TLB0: 4M pages, 4-way, 16 entries";
    break;
  case 0x57:
    std::cout << "Data TLB0: 4K pages, 4-way, 16 entries";
    break;
  case 0x59:
    std::cout << "Data TLB0: 4K pages, fully associative, 16 entries";
    break;
  case 0x5A:
    std::cout << "Data TLB0: 2M/4M pages, 4-way, 32 entries";
    break;
  case 0x5B:
    std::cout << "Data TLB: 4K/4M pages, 64 entries";
    break;
  case 0x5C:
    std::cout << "Data TLB: 4K/4M pages, 128 entries";
    break;
  case 0x5D:
    std::cout << "Data TLB: 4K/4M pages, 256 entries";
    break;
  case 0x60:
    std::cout << "1st-level data cache: 16KB, 8-way, 64 byte line size";
    break;
  case 0x61:
    std::cout << "Instruction TLB: 4K pages, fully associative, 48 entries";
    break;
  case 0x63:
    std::cout << "Data TLB: 2M/4M pages, 4-way, 32 entries and 1GB pages, 4-way, 4 entries";
    break;
  case 0x64:
    std::cout << "Data TLB: 4K pages, 4-way, 512 entries";
    break;
  case 0x66:
    std::cout << "1st-level data cache: 8KB, 4-way, 64 byte line size";
    break;
  case 0x67:
    std::cout << "1st-level data cache: 16KB, 4-way, 64 byte line size";
    break;
  case 0x68:
    std::cout << "1st-level data cache: 32KB, 4-way, 64 byte line size";
    break;
  case 0x6A:
    std::cout << "uTLB: 4K pages, 8-way, 64 entries";
    break;
  case 0x6B:
    std::cout << "DTLB: 4K pages, 8-way, 256 entries";
    break;
  case 0x6C:
    std::cout << "DTLB: 2M/4M pages, 8-way, 128 entries";
    break;
  case 0x6D:
    std::cout << "DTLB: 1GB pages, fully associative, 16 entries";
    break;
  case 0x70:
    std::cout << "Trace cache: 12K-μop, 8-way";
    break;
  case 0x71:
    std::cout << "Trace cache: 16K-μop, 8-way";
    break;
  case 0x72:
    std::cout << "Trace cache: 32K-μop, 8-way";
    break;
  case 0x76:
    std::cout << "Instruction TLB: 2M/4M pages, fully associative, 8 entries";
    break;
  case 0x78:
    std::cout << "2nd-level cache: 1MB, 4-way, 64 byte line size";
    break;
  case 0x79:
    std::cout << "2nd-level cache: 128KB, 8-way, 64 byte line size, 2 lines per sector";
    break;
  case 0x7A:
    std::cout << "2nd-level cache: 256KB, 8-way, 64 byte line size, 2 lines per sector";
    break;
  case 0x7B:
    std::cout << "2nd-level cache: 512KB, 8-way, 64 byte line size, 2 lines per sector";
    break;
  case 0x7C:
    std::cout << "2nd-level cache: 1MB, 8-way, 64 byte line size, 2 lines per sector";
    break;
  case 0x7D:
    std::cout << "2nd-level cache: 2MB, 8-way, 64 byte line size";
    break;
  case 0x7F:
    std::cout << "2nd-level cache: 512KB, 2-way, 64 byte line size";
    break;
  case 0x80:
    std::cout << "2nd-level cache: 512KB, 8-way, 64 byte line size";
    break;
  case 0x82:
    std::cout << "2nd-level cache: 256KB, 8-way, 32 byte line size";
    break;
  case 0x83:
    std::cout << "2nd-level cache: 512KB, 8-way, 32 byte line size";
    break;
  case 0x84:
    std::cout << "2nd-level cache: 1MB, 8-way, 32 byte line size";
    break;
  case 0x85:
    std::cout << "2nd-level cache: 2MB, 8-way, 32 byte line size";
    break;
  case 0x86:
    std::cout << "2nd-level cache: 512KB, 4-way, 64 byte line size";
    break;
  case 0x87:
    std::cout << "2nd-level cache: 1MB, 8-way, 64 byte line size";
    break;
  case 0xA0:
    std::cout << "DTLB: 4k pages, fully associative, 32 entries";
    break;
  case 0xB0:
    std::cout << "Instruction TLB: 4 KByte pages, 4-way set associative, 128 entries";
    break;
  case 0xB1:
    std::cout << "Instruction TLB: 2M pages, 4-way, 8 entries or 4M pages, 4-way, 4 entries";
    break;
  case 0xB2:
    std::cout << "Instruction TLB: 4KByte pages, 4-way set associative, 64 entries";
    break;
  case 0xB3:
    std::cout << "Data TLB: 4 KByte pages, 4-way set associative, 128 entries";
    break;
  case 0xB4:
    std::cout << "Data TLB: 4M pages, 4-way set associative, 8 entries";
    break;
  case 0xB5:
    std::cout << "Data TLB1: 4M pages, 4-way associative, 256 entries";
    break;
  case 0xB6:
    std::cout << "Instruction TLB: 4KByte pages, 8-way set associative, 64 entries";
    break;
  case 0xB7:
    std::cout << "Instruction TLB: 2M pages, 8-way, 2 entries or 4M pages, 8-way, 1 entry";
    break;
  case 0xB8:
    std::cout << "Instruction TLB: 4KByte pages, 8-way set associative, 128 entries";
    break;
  case 0xBA:
    std::cout << "Data TLB1: 4 KByte pages, 4-way associative, 64 entries";
    break;
  case 0xC0:
    std::cout << "Data TLB: 4 KByte and 4 MByte pages, 4-way associative, 8 entries";
    break;
  case 0xC1:
    std::cout << "Shared 2nd-Level TLB: 4 KByte/2MByte pages, 8-way associative, 1024 entries";
    break;
  case 0xC2:
    std::cout << "DTLB: 4 KByte/2 MByte pages, 4-way associative, 16 entries";
    break;
  case 0xC3:
    std::cout << "Shared 2nd-Level TLB: 4 KByte/2 MByte pages, 6-way associative, 1536 entries";
    break;
  case 0xC4:
    std::cout << "DTLB: 2M/4M Byte pages, 4-way associative, 32 entries";
    break;
  case 0xCA:
    std::cout << "Shared 2nd-Level TLB: 4 KByte pages, 4-way associative, 512 entries";
    break;
  case 0xD0:
    std::cout << "3rd-level cache: 512 KByte, 4-way set associative, 64 byte line size";
    break;
  case 0xD1:
    std::cout << "3rd-level cache: 1 MByte, 4-way set associative, 64 byte line size";
    break;
  case 0xD2:
    std::cout << "3rd-level cache: 2 MByte, 4-way set associative, 64 byte line size";
    break;
  case 0xD6:
    std::cout << "3rd-level cache: 1 MByte, 8-way set associative, 64 byte line size";
    break;
  case 0xD7:
    std::cout << "3rd-level cache: 2 MByte, 8-way set associative, 64 byte line size";
    break;
  case 0xD8:
    std::cout << "3rd-level cache: 4 MByte, 8-way set associative, 64 byte line size";
    break;
  case 0xDC:
    std::cout << "3rd-level cache: 1.5 MByte, 12-way set associative, 64 byte line size";
    break;
  case 0xDD:
    std::cout << "3rd-level cache: 3 MByte, 12-way set associative, 64 byte line size";
    break;
  case 0xDE:
    std::cout << "3rd-level cache: 6 MByte, 12-way set associative, 64 byte line size";
    break;
  case 0xE2:
    std::cout << "3rd-level cache: 2 MByte, 16-way set associative, 64 byte line size";
    break;
  case 0xE3:
    std::cout << "3rd-level cache: 4 MByte, 16-way set associative, 64 byte line size";
    break;
  case 0xE4:
    std::cout << "3rd-level cache: 8 MByte, 16-way set associative, 64 byte line size";
    break;
  case 0xEA:
    std::cout << "3rd-level cache: 12 MByte, 24-way set associative, 64 byte line size";
    break;
  case 0xEB:
    std::cout << "3rd-level cache: 18 MByte, 24-way set associative, 64 byte line size";
    break;
  case 0xEC:
    std::cout << "3rd-level cache: 24 MByte, 24-way set associative, 64 byte line size";
    break;
  case 0xF0:
    std::cout << "Prefetch: 64-Byte prefetching";
    break;
  case 0xF1:
    std::cout << "Prefetch: 128-Byte prefetching";
    break;
  case 0xFE:
    std::cout << "CPUID leaf 2 does not report TLB descriptor information";
    break;
  case 0xFF:
    std::cout << "CPUID leaf 2 does not report cache descriptor information";
    break;
  default:
    std::cout << "Unknown descriptor";
    break;
  }
  std::cout << std::endl;
}

void printTlbLeaf0x2Descriptor(unsigned int eax, unsigned int ebx, unsigned int ecx,
                               unsigned int edx) {
  unsigned char descriptors[15] = {static_cast<unsigned char>((eax >> 8) & 0xFF),
                                   static_cast<unsigned char>((eax >> 16) & 0xFF),
                                   static_cast<unsigned char>((eax >> 24) & 0xFF),
                                   static_cast<unsigned char>(ebx & 0xFF),
                                   static_cast<unsigned char>((ebx >> 8) & 0xFF),
                                   static_cast<unsigned char>((ebx >> 16) & 0xFF),
                                   static_cast<unsigned char>((ebx >> 24) & 0xFF),
                                   static_cast<unsigned char>(ecx & 0xFF),
                                   static_cast<unsigned char>((ecx >> 8) & 0xFF),
                                   static_cast<unsigned char>((ecx >> 16) & 0xFF),
                                   static_cast<unsigned char>((ecx >> 24) & 0xFF),
                                   static_cast<unsigned char>(edx & 0xFF),
                                   static_cast<unsigned char>((edx >> 8) & 0xFF),
                                   static_cast<unsigned char>((edx >> 16) & 0xFF),
                                   static_cast<unsigned char>((edx >> 24) & 0xFF)};

  std::cout << "Byte 0: Reserved (ignored)" << std::endl;
  for(int i = 0; i < 15; i++) {
    std::cout << "Byte " << i + 1 << ": ";
    printByteDescriptorFromLeaf0x2(descriptors[i]);
  }
}

bool has0xfeByteValue(unsigned int reg, unsigned char byteValue) {
  for(int i = 0; i < 4; i++) {
    if((reg & 0xFF) == byteValue) {
      return true;
    }
    reg >>= 8;
  }
  return false;
}

void printIntelTlbSpecifications() {
  unsigned int eax, ebx, ecx, edx;
  eax = 0x02;

  asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax));

  unsigned char byteValue = 0xFE;
  if(has0xfeByteValue(eax, byteValue) || has0xfeByteValue(ebx, byteValue) ||
     has0xfeByteValue(ecx, byteValue) || has0xfeByteValue(edx, byteValue)) {

    eax = 0x18;
    ecx = 0x0;

    asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));

    printTlbSpecificationsFromLeaf0x18(eax);
    return;
  }

  printf("0x%08x-", eax);
  printf("0x%08x-", ebx);
  printf("0x%08x-", ecx);
  printf("0x%08x", edx);

  std::cout << std::endl;

  std::string eax_binary = std::bitset<32>(eax).to_string();
  std::string ebx_binary = std::bitset<32>(ebx).to_string();
  std::string ecx_binary = std::bitset<32>(ecx).to_string();
  std::string edx_binary = std::bitset<32>(edx).to_string();

  for(int i = 4; i < 39; i += 5) {
    eax_binary.insert(i, " ");
    ebx_binary.insert(i, " ");
    ecx_binary.insert(i, " ");
    edx_binary.insert(i, " ");
  }

  std::cout << eax_binary << "---";
  std::cout << ebx_binary << "---";
  std::cout << ecx_binary << "---";
  std::cout << edx_binary << std::endl;

  printTlbLeaf0x2Descriptor(eax, ebx, ecx, edx);
}

uint32_t getL2TlbEntriesFor4KbytePages0x2LeafDescriptor(unsigned char value) {
  switch(value) {
  case 0xC1:
    return 1024;
  case 0xC3:
    return 1536;
  case 0xCA:
    return 512;
  default:
    return 0;
  }
}

uint32_t getL2TlbEntriesFor4KbytePages0x2Leaf(unsigned int eax, unsigned int ebx, unsigned int ecx,
                                              unsigned int edx) {
  unsigned char descriptors[15] = {static_cast<unsigned char>((eax >> 8) & 0xFF),
                                   static_cast<unsigned char>((eax >> 16) & 0xFF),
                                   static_cast<unsigned char>((eax >> 24) & 0xFF),
                                   static_cast<unsigned char>(ebx & 0xFF),
                                   static_cast<unsigned char>((ebx >> 8) & 0xFF),
                                   static_cast<unsigned char>((ebx >> 16) & 0xFF),
                                   static_cast<unsigned char>((ebx >> 24) & 0xFF),
                                   static_cast<unsigned char>(ecx & 0xFF),
                                   static_cast<unsigned char>((ecx >> 8) & 0xFF),
                                   static_cast<unsigned char>((ecx >> 16) & 0xFF),
                                   static_cast<unsigned char>((ecx >> 24) & 0xFF),
                                   static_cast<unsigned char>(edx & 0xFF),
                                   static_cast<unsigned char>((edx >> 8) & 0xFF),
                                   static_cast<unsigned char>((edx >> 16) & 0xFF),
                                   static_cast<unsigned char>((edx >> 24) & 0xFF)};

  uint32_t totalL2Tlb4KbytePages = 0;
  for(unsigned char descriptor : descriptors) {
    totalL2Tlb4KbytePages += getL2TlbEntriesFor4KbytePages0x2LeafDescriptor(descriptor);
  }
  return totalL2Tlb4KbytePages;
}

uint32_t getL2TlbEntriesFor4KbytePages0x18Leaf(unsigned int maxSubleaf) {
  unsigned int leaf = 0x18;
  unsigned int eax, ebx, ecx, edx;

  uint32_t totalL2Tlb4KbytePages = 0;

  for(unsigned int subleaf = 0x1; subleaf <= maxSubleaf; subleaf++) {
    eax = leaf;
    ecx = subleaf;

    asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));

    unsigned int translationCacheLevel = (edx >> 5) & 0x7;
    unsigned int translationCacheType = edx & 0x1F;
    unsigned int waysOfAssociativity = (ebx >> 16) & 0xFF;
    unsigned int numberOfSets = ecx;
    bool supports4KPage = ebx & 0x1;

    if(translationCacheLevel == 2 &&
       (translationCacheType == 1 || translationCacheType == 3 || translationCacheType == 5) &&
       supports4KPage) {
      totalL2Tlb4KbytePages +=
          static_cast<uint32_t>(waysOfAssociativity) * static_cast<uint32_t>(numberOfSets);
    }
  }

  return totalL2Tlb4KbytePages;
}

uint32_t l2TlbEntriesFor4KbytePages() {
  unsigned int eax, ebx, ecx, edx;
  eax = 0x02;

  asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax));

  unsigned char byteValue = 0xFE;
  if(has0xfeByteValue(eax, byteValue) || has0xfeByteValue(ebx, byteValue) ||
     has0xfeByteValue(ecx, byteValue) || has0xfeByteValue(edx, byteValue)) {

    eax = 0x18;
    ecx = 0x0;

    asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));

    return getL2TlbEntriesFor4KbytePages0x18Leaf(eax);
  }

  return getL2TlbEntriesFor4KbytePages0x2Leaf(eax, ebx, ecx, edx);
}

} // namespace adaptive
