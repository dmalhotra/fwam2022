CXX=g++ # requires g++-8 or newer / icpc (with gcc compatibility 7.5 or newer) / clang++ with llvm-10 or newer
CXXFLAGS = -O3 -march=native -std=c++11 -fopenmp # need C++11 and OpenMP

RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./SCTL/include

TARGET_BIN = \
       $(BINDIR)/instruction-cost \
       $(BINDIR)/poly-eval \
       $(BINDIR)/gemm-ker \
       $(BINDIR)/gemm-blocking \
       $(BINDIR)/bandwidth-l1 \
       $(BINDIR)/bandwidth-main-memory

all : $(TARGET_BIN)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $^ $(CXXFLAGS) $(LDLIBS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $^ -o $@

.PHONY: all check clean

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~ */*/*~
