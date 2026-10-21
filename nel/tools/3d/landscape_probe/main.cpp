// landscape_probe - offline height measurement for NeL landscape data.
//
// Measures, at given world X/Y, the rendered terrain height (bezier patches of
// .zone/.zonel files), the PACS collision height (retriever bank + global
// retriever) and the foundation height of instance-group instances
// (instance position combined with the shape's bounding box). Nothing here
// renders or changes data; it exists to establish whether buildings float and
// by how much, before any correction is chosen.
//
// Usage:
//   landscape_probe --zone <file>... [--rbank <dir> --bank <name>]
//                   [--ig <file>...] [--shapes <dir>...]
//                   [--at x,y]... [--instances] [--offset dx,dy]
//                   [--step <metres>]
//
//   --points     file with one "x,y" per line (same output as --at)
//   --at         probe a world point: terrain z (nearest tessellated sample),
//                PACS z (mean height of the retrieved surface)
//   --instances  for every instance of every --ig: pivot z, foundation z
//                (lowest transformed bbox corner), terrain z under the pivot,
//                PACS z under the pivot, and the differences
//   --offset     added to every instance/point before sampling the zones
//                (lets a library brick at the origin be compared with its
//                installed, translated copy)
//   --step       tessellation sample spacing in metres (default 0.25)

#include "nel/misc/types_nl.h"
#include "nel/misc/app_context.h"
#include "nel/misc/file.h"
#include "nel/misc/path.h"
#include "nel/misc/aabbox.h"
#include "nel/misc/matrix.h"
#include "nel/3d/zone.h"
#include "nel/3d/scene_group.h"
#include "nel/3d/shape.h"
#include "nel/3d/register_3d.h"
#include "nel/pacs/u_retriever_bank.h"
#include "nel/pacs/u_global_retriever.h"
#include "nel/pacs/u_global_position.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <cmath>

using namespace NLMISC;
using namespace NL3D;
using namespace NLPACS;

namespace {

struct CSample
{
	float X, Y, Z;
	uint16 Zone;
	uint16 Patch;
};

struct CTerrain
{
	std::vector<CSample> Samples;
	// coarse grid for lookup
	std::map<std::pair<sint32, sint32>, std::vector<uint32> > Grid;
	float Cell;

	CTerrain() : Cell(4.0f) {}

	void addZone(const std::string &path, float step)
	{
		CIFile f;
		if (!f.open(path)) { fprintf(stderr, "cannot open zone %s\n", path.c_str()); exit(2); }
		CZone zone;
		f.serial(zone);
		CZoneInfo zi;
		zone.retrieve(zi);
		for (uint p = 0; p < zi.Patchs.size(); ++p)
		{
			const CPatchInfo &pi = zi.Patchs[p];
			// patch edge length is about 2 m per tile; sample so that spacing <= step
			uint ns = std::max<uint>(2, (uint)ceil(pi.OrderS * 2.0f / step));
			uint nt = std::max<uint>(2, (uint)ceil(pi.OrderT * 2.0f / step));
			for (uint i = 0; i <= ns; ++i)
				for (uint j = 0; j <= nt; ++j)
				{
					CVector v = pi.Patch.eval((float)i / ns, (float)j / nt);
					CSample s; s.X = v.x; s.Y = v.y; s.Z = v.z; s.Zone = zone.getZoneId(); s.Patch = (uint16)p;
					uint32 idx = (uint32)Samples.size();
					Samples.push_back(s);
					Grid[std::make_pair((sint32)floor(v.x / Cell), (sint32)floor(v.y / Cell))].push_back(idx);
				}
		}
		fprintf(stderr, "zone %s: %u patches, %u samples so far\n", CFile::getFilename(path).c_str(), (uint)zi.Patchs.size(), (uint)Samples.size());
	}

	// nearest sample in XY; returns false if none within maxDist
	bool nearest(float x, float y, float maxDist, CSample &out, float &dist) const
	{
		float best = maxDist * maxDist;
		bool found = false;
		sint32 cx = (sint32)floor(x / Cell), cy = (sint32)floor(y / Cell);
		sint32 r = (sint32)ceil(maxDist / Cell);
		for (sint32 gx = cx - r; gx <= cx + r; ++gx)
			for (sint32 gy = cy - r; gy <= cy + r; ++gy)
			{
				std::map<std::pair<sint32, sint32>, std::vector<uint32> >::const_iterator it = Grid.find(std::make_pair(gx, gy));
				if (it == Grid.end()) continue;
				for (uint k = 0; k < it->second.size(); ++k)
				{
					const CSample &s = Samples[it->second[k]];
					float d = (s.X - x) * (s.X - x) + (s.Y - y) * (s.Y - y);
					if (d < best) { best = d; out = s; found = true; }
				}
			}
		dist = sqrtf(best);
		return found;
	}
};

struct CPacs
{
	URetrieverBank *Bank;
	UGlobalRetriever *GR;
	CPacs() : Bank(NULL), GR(NULL) {}

	bool init(const std::string &dir, const std::string &name)
	{
		CPath::addSearchPath(dir, false, false);
		Bank = URetrieverBank::createRetrieverBank((name + ".rbank").c_str(), true);
		if (!Bank) { fprintf(stderr, "cannot load %s.rbank from %s\n", name.c_str(), dir.c_str()); return false; }
		GR = UGlobalRetriever::createGlobalRetriever((name + ".gr").c_str(), Bank);
		if (!GR) { fprintf(stderr, "cannot load %s.gr\n", name.c_str()); return false; }
		return true;
	}

	// returns true with z = mean height of the surface retrieved around (x,y,zEstimate)
	bool height(float x, float y, float zEstimate, float &z, sint32 &instance, sint32 &surface) const
	{
		if (!GR) return false;
		UGlobalPosition gp = GR->retrievePosition(CVectorD(x, y, zEstimate));
		instance = gp.InstanceId;
		surface = gp.LocalPosition.Surface;
		if (gp.InstanceId < 0) return false;
		z = GR->getMeanHeight(gp);
		return true;
	}
};

struct CShapeCache
{
	std::map<std::string, CAABBox> Boxes;
	std::map<std::string, bool> Missing;

	bool box(const std::string &name, CAABBox &bb)
	{
		std::map<std::string, CAABBox>::const_iterator it = Boxes.find(name);
		if (it != Boxes.end()) { bb = it->second; return true; }
		if (Missing.count(name)) return false;
		std::string file = name;
		if (CFile::getExtension(file).empty()) file += ".shape";
		std::string path = CPath::lookup(file, false, false);
		if (path.empty()) { Missing[name] = true; return false; }
		try
		{
			CIFile f;
			if (!f.open(path)) { Missing[name] = true; return false; }
			CShapeStream ss;
			f.serial(ss);
			IShape *shape = ss.getShapePointer();
			if (!shape) { Missing[name] = true; return false; }
			shape->getAABBox(bb);
			delete shape;
			Boxes[name] = bb;
			return true;
		}
		catch (const Exception &e)
		{
			fprintf(stderr, "shape %s: %s\n", name.c_str(), e.what());
			Missing[name] = true;
			return false;
		}
	}
};

bool parseXY(const char *s, float &x, float &y)
{
	return sscanf(s, "%f,%f", &x, &y) == 2;
}

} // namespace

int main(int argc, char **argv)
{
	CApplicationContext appContext;
	registerSerial3d();

	std::vector<std::string> zones, igs;
	std::string rbankDir, bankName;
	std::vector<std::pair<float, float> > points;
	bool listInstances = false;
	float offX = 0, offY = 0, step = 0.25f;

	for (int i = 1; i < argc; ++i)
	{
		std::string a = argv[i];
		bool more = (i + 1 < argc);
		if (a == "--zone" && more) zones.push_back(argv[++i]);
		else if (a == "--ig" && more) igs.push_back(argv[++i]);
		else if (a == "--rbank" && more) rbankDir = argv[++i];
		else if (a == "--bank" && more) bankName = argv[++i];
		else if (a == "--shapes" && more) CPath::addSearchPath(argv[++i], false, false);
		else if (a == "--at" && more) { float x, y; if (!parseXY(argv[++i], x, y)) { fprintf(stderr, "bad --at\n"); return 1; } points.push_back(std::make_pair(x, y)); }
		else if (a == "--points" && more)
		{
			// one "x,y" per line; lets callers probe a whole grid without a huge command line
			FILE *pf = fopen(argv[++i], "r");
			if (!pf) { fprintf(stderr, "cannot open points file\n"); return 1; }
			char line[256];
			while (fgets(line, sizeof(line), pf)) { float x, y; if (parseXY(line, x, y)) points.push_back(std::make_pair(x, y)); }
			fclose(pf);
		}
		else if (a == "--offset" && more) { if (!parseXY(argv[++i], offX, offY)) { fprintf(stderr, "bad --offset\n"); return 1; } }
		else if (a == "--step" && more) step = (float)atof(argv[++i]);
		else if (a == "--instances") listInstances = true;
		else { fprintf(stderr, "unknown or incomplete argument: %s\n", a.c_str()); return 1; }
	}
	if (zones.empty()) { fprintf(stderr, "need at least one --zone\n"); return 1; }

	CTerrain terrain;
	for (uint i = 0; i < zones.size(); ++i) terrain.addZone(zones[i], step);

	CPacs pacs;
	bool havePacs = false;
	if (!rbankDir.empty() && !bankName.empty()) havePacs = pacs.init(rbankDir, bankName);

	const float maxDist = 2.0f;

	if (!points.empty())
	{
		printf("point\tx\ty\tterrain_z\tsample_dist\tzone\tpatch\tpacs_z\tpacs_instance\tpacs_surface\tpacs_minus_terrain\n");
		for (uint i = 0; i < points.size(); ++i)
		{
			float x = points[i].first + offX, y = points[i].second + offY;
			CSample s; float d;
			bool ok = terrain.nearest(x, y, maxDist, s, d);
			float pz = 0; sint32 inst = -1, surf = -1; bool pok = false;
			if (ok && havePacs) pok = pacs.height(x, y, s.Z, pz, inst, surf);
			printf("%u\t%.2f\t%.2f\t", i, x, y);
			if (ok) printf("%.3f\t%.2f\t%u\t%u\t", s.Z, d, s.Zone, s.Patch); else printf("none\t-\t-\t-\t");
			if (pok) printf("%.3f\t%d\t%d\t%.3f\n", pz, inst, surf, pz - s.Z); else printf("none\t%d\t%d\t-\n", inst, surf);
		}
	}

	if (listInstances)
	{
		CShapeCache shapes;
		printf("ig\tinstance\tshape\tx\ty\tpivot_z\tfoundation_z\tbbox_min_z\tterrain_z\tsample_dist\tfoundation_minus_terrain\tpivot_minus_terrain\tpacs_z\tpacs_instance\tpacs_minus_terrain\n");
		for (uint g = 0; g < igs.size(); ++g)
		{
			CIFile f;
			if (!f.open(igs[g])) { fprintf(stderr, "cannot open ig %s\n", igs[g].c_str()); continue; }
			CInstanceGroup ig;
			f.serial(ig);
			CVector gpos = ig.getGlobalPos();
			std::string igName = CFile::getFilename(igs[g]);
			for (uint k = 0; k < ig._InstancesInfos.size(); ++k)
			{
				const CInstanceGroup::CInstance &in = ig._InstancesInfos[k];
				CVector pos = in.Pos + gpos;
				float x = pos.x + offX, y = pos.y + offY;
				CAABBox bb; bool hasBox = shapes.box(in.Name, bb);
				float foundation = pos.z, bbMinZ = 0;
				if (hasBox)
				{
					CMatrix m; m.identity(); m.setRot(in.Rot); m.setPos(pos); m.scale(in.Scale);
					CVector mn = bb.getMin(), mx = bb.getMax();
					float lowest = 1e30f;
					for (int c = 0; c < 8; ++c)
					{
						CVector corner((c & 1) ? mx.x : mn.x, (c & 2) ? mx.y : mn.y, (c & 4) ? mx.z : mn.z);
						CVector w = m * corner;
						if (w.z < lowest) lowest = w.z;
					}
					foundation = lowest;
					bbMinZ = mn.z;
				}
				CSample s; float d;
				bool ok = terrain.nearest(x, y, maxDist, s, d);
				float pz = 0; sint32 inst = -1, surf = -1; bool pok = false;
				if (ok && havePacs) pok = pacs.height(x, y, s.Z, pz, inst, surf);
				printf("%s\t%s\t%s\t%.2f\t%.2f\t%.3f\t", igName.c_str(), in.InstanceName.c_str(), in.Name.c_str(), x, y, pos.z);
				if (hasBox) printf("%.3f\t%.3f\t", foundation, bbMinZ); else printf("noshape\t-\t");
				if (ok) printf("%.3f\t%.2f\t", s.Z, d); else printf("none\t-\t");
				if (ok && hasBox) printf("%.3f\t", foundation - s.Z); else printf("-\t");
				if (ok) printf("%.3f\t", pos.z - s.Z); else printf("-\t");
				if (pok) printf("%.3f\t%d\t%.3f\n", pz, inst, pz - s.Z); else printf("none\t%d\t-\n", inst);
			}
		}
	}
	return 0;
}
