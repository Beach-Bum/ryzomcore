// zone_from_heightfield - build a native NeL zone whose patches are fitted to a height grid.
//
// Why this exists (engine-test #40): the pipeline's zone_elevation displaces an existing brick's
// patches by moving their four corners and rotating the tangents. A jungle 1x1 brick has 5x5
// patches of 32 m, so everything finer than 32 m is lost - a six-area landscape measured 4.66 m
// off its Blender source, and a flat 20 m clearing came out with 0.86 m of relief.
//
// A NeL patch is a bicubic Bezier with OrderS/OrderT tiles of 2 m, OrderS <= 16. Nothing stops a
// zone from being built out of smaller patches: 160 m of order 4 is 20x20 patches of 8 m. This
// tool writes such a zone directly, with every control point fitted to the height grid
// (corner value, edge tangents from the first derivative, interiors from the cross derivative),
// which is C1 across patch borders and interpolates the grid at every patch corner.
//
// Tile material, lumel and light-influence layout are copied from a template zone, so the result
// stays loadable by the shipped tools and by the client's tile bank.
//
// Usage:
//   zone_from_heightfield --template <in.zonenh> --heights <f32> --grid 257 --cell-size 160
//                         --origin <x>,<y> --order 4 --out <out.zone> [--zone-name 84_JT]
//                         [--report <csv>]
//
//   --heights   little-endian float32, rows ascending Y, columns ascending X (grid x grid)
//   --origin    world X,Y of the cell's minimum corner
//   --order     tiles per patch edge: 2, 4, 8 or 16 (patch edge = order * 2 m)
//   --report    write "x,y,height_source,height_patch" for every grid point

#include "nel/misc/types_nl.h"
#include "nel/misc/app_context.h"
#include "nel/misc/file.h"
#include "nel/misc/path.h"
#include "nel/misc/vector.h"
#include "nel/3d/zone.h"
#include "nel/3d/register_3d.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

using namespace NLMISC;
using namespace NL3D;

namespace {

// Catmull-Rom sampling of the height grid: smooth, interpolating, and with analytic derivatives
// so the Bezier tangents describe the same surface the grid does.
struct CHeightField
{
	std::vector<float> Z;
	uint N;
	float Step;   // metres between samples
	float Size;   // metres across

	bool load(const std::string &path, uint grid, float size)
	{
		N = grid; Size = size; Step = size / (float)(grid - 1);
		FILE *f = fopen(path.c_str(), "rb");
		if (!f) return false;
		Z.resize((size_t)N * N);
		size_t got = fread(&Z[0], sizeof(float), Z.size(), f);
		fclose(f);
		return got == Z.size();
	}

	float at(sint ix, sint iy) const
	{
		ix = std::max<sint>(0, std::min<sint>((sint)N - 1, ix));
		iy = std::max<sint>(0, std::min<sint>((sint)N - 1, iy));
		return Z[(size_t)iy * N + ix];
	}

	static void crWeights(float t, float w[4], float d[4])
	{
		float t2 = t * t, t3 = t2 * t;
		w[0] = 0.5f * (-t3 + 2.f * t2 - t);
		w[1] = 0.5f * (3.f * t3 - 5.f * t2 + 2.f);
		w[2] = 0.5f * (-3.f * t3 + 4.f * t2 + t);
		w[3] = 0.5f * (t3 - t2);
		d[0] = 0.5f * (-3.f * t2 + 4.f * t - 1.f);
		d[1] = 0.5f * (9.f * t2 - 10.f * t);
		d[2] = 0.5f * (-9.f * t2 + 8.f * t + 1.f);
		d[3] = 0.5f * (3.f * t2 - 2.f * t);
	}

	// local x,y in metres from the cell's minimum corner
	void sample(float x, float y, float &f, float &fx, float &fy, float &fxy) const
	{
		float gx = x / Step, gy = y / Step;
		sint ix = (sint)floorf(gx), iy = (sint)floorf(gy);
		float tx = gx - (float)ix, ty = gy - (float)iy;
		float wx[4], dx[4], wy[4], dy[4];
		crWeights(tx, wx, dx);
		crWeights(ty, wy, dy);
		f = fx = fy = fxy = 0.f;
		for (sint j = 0; j < 4; ++j)
			for (sint i = 0; i < 4; ++i)
			{
				float v = at(ix - 1 + i, iy - 1 + j);
				f   += v * wx[i] * wy[j];
				fx  += v * dx[i] * wy[j];
				fy  += v * wx[i] * dy[j];
				fxy += v * dx[i] * dy[j];
			}
		fx /= Step; fy /= Step; fxy /= (Step * Step);
	}
};

bool parseXY(const char *s, float &x, float &y) { return sscanf(s, "%f,%f", &x, &y) == 2; }

// NeL's control grid, from CBezierPatch::eval: C[i][j], i along s, j along t.
struct CCtrl { CVector C[4][4]; };

} // namespace

int main(int argc, char **argv)
{
	CApplicationContext appContext;
	registerSerial3d();

	std::string tpl, heights, out, report, zoneName;
	uint grid = 257, order = 4;
	float cellSize = 160.f, ox = 0.f, oy = 0.f;

	for (int i = 1; i < argc; ++i)
	{
		std::string a = argv[i];
		bool more = (i + 1 < argc);
		if (a == "--template" && more) tpl = argv[++i];
		else if (a == "--heights" && more) heights = argv[++i];
		else if (a == "--out" && more) out = argv[++i];
		else if (a == "--report" && more) report = argv[++i];
		else if (a == "--zone-name" && more) zoneName = argv[++i];
		else if (a == "--grid" && more) grid = (uint)atoi(argv[++i]);
		else if (a == "--order" && more) order = (uint)atoi(argv[++i]);
		else if (a == "--cell-size" && more) cellSize = (float)atof(argv[++i]);
		else if (a == "--origin" && more) { if (!parseXY(argv[++i], ox, oy)) { fprintf(stderr, "bad --origin\n"); return 1; } }
		else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 1; }
	}
	if (tpl.empty() || heights.empty() || out.empty()) { fprintf(stderr, "need --template, --heights and --out\n"); return 1; }
	if (order != 2 && order != 4 && order != 8 && order != 16) { fprintf(stderr, "--order must be 2, 4, 8 or 16\n"); return 1; }

	float patchSize = (float)order * 2.f;               // a tile is 2 m
	if (fmodf(cellSize, patchSize) != 0.f) { fprintf(stderr, "cell size %.1f is not a multiple of the patch size %.1f\n", cellSize, patchSize); return 1; }
	uint P = (uint)(cellSize / patchSize);              // patches per side

	CHeightField hf;
	if (!hf.load(heights, grid, cellSize)) { fprintf(stderr, "cannot read %u*%u floats from %s\n", grid, grid, heights.c_str()); return 1; }

	// Template: tile material, lumel and light-influence layout, and the zone id when not named.
	CZone tplZone;
	{
		CIFile f;
		if (!f.open(tpl)) { fprintf(stderr, "cannot open template %s\n", tpl.c_str()); return 1; }
		f.serial(tplZone);
	}
	CZoneInfo tplInfo;
	tplZone.retrieve(tplInfo);
	if (tplInfo.Patchs.empty()) { fprintf(stderr, "template has no patch\n"); return 1; }
	const CPatchInfo &src = tplInfo.Patchs[0];
	uint srcTiles = (uint)src.OrderS * src.OrderT;
	if (srcTiles == 0 || src.Tiles.empty()) { fprintf(stderr, "template patch has no tile\n"); return 1; }
	uint lumelPerTile = (uint)(src.Lumels.size() / srcTiles);
	uint liPerCorner = src.TileLightInfluences.empty() ? 0
		: (uint)(src.TileLightInfluences.size() / (((uint)src.OrderS + 1) * ((uint)src.OrderT + 1)));

	// A cell holds the same number of 2 m tiles whatever the patch order, so the template's tiles
	// can be laid out once as a grid and read back per generated patch. Copying one tile everywhere
	// instead produced a zone PACS tessellated to nothing: the retriever came out empty at every
	// order tried, which is what identified this as a tile problem rather than a patch-size one.
	uint tilesPerSide = (uint)(cellSize / 2.f);
	std::vector<CTileElement> tileGrid((size_t)tilesPerSide * tilesPerSide, src.Tiles[0]);
	std::vector<CTileColor> colorGrid(((size_t)tilesPerSide + 1) * (tilesPerSide + 1), src.TileColors.empty() ? CTileColor() : src.TileColors[0]);
	{
		float tplMinX = 1e30f, tplMinY = 1e30f;
		for (uint p = 0; p < tplInfo.Patchs.size(); ++p)
			for (uint v = 0; v < 4; ++v)
			{
				tplMinX = std::min(tplMinX, tplInfo.Patchs[p].Patch.Vertices[v].x);
				tplMinY = std::min(tplMinY, tplInfo.Patchs[p].Patch.Vertices[v].y);
			}
		uint placed = 0;
		for (uint p = 0; p < tplInfo.Patchs.size(); ++p)
		{
			const CPatchInfo &tp = tplInfo.Patchs[p];
			if (tp.Tiles.size() != (size_t)tp.OrderS * tp.OrderT) continue;
			float pminx = 1e30f, pminy = 1e30f;
			for (uint v = 0; v < 4; ++v) { pminx = std::min(pminx, tp.Patch.Vertices[v].x); pminy = std::min(pminy, tp.Patch.Vertices[v].y); }
			sint tx0 = (sint)floorf((pminx - tplMinX) / 2.f + 0.5f);
			sint ty0 = (sint)floorf((pminy - tplMinY) / 2.f + 0.5f);
			bool sIsX = fabsf(tp.Patch.Vertices[3].x - tp.Patch.Vertices[0].x) > fabsf(tp.Patch.Vertices[3].y - tp.Patch.Vertices[0].y);
			for (uint tt = 0; tt < (uint)tp.OrderT; ++tt)
				for (uint ts = 0; ts < (uint)tp.OrderS; ++ts)
				{
					sint gx = tx0 + (sint)(sIsX ? ts : tt);
					sint gy = ty0 + (sint)(sIsX ? tt : ts);
					if (gx < 0 || gy < 0 || gx >= (sint)tilesPerSide || gy >= (sint)tilesPerSide) continue;
					tileGrid[(size_t)gy * tilesPerSide + gx] = tp.Tiles[ts + tt * tp.OrderS];
					++placed;
				}
			if (!tp.TileColors.empty())
				for (uint tt = 0; tt <= (uint)tp.OrderT; ++tt)
					for (uint ts = 0; ts <= (uint)tp.OrderS; ++ts)
					{
						sint gx = tx0 + (sint)(sIsX ? ts : tt);
						sint gy = ty0 + (sint)(sIsX ? tt : ts);
						if (gx < 0 || gy < 0 || gx > (sint)tilesPerSide || gy > (sint)tilesPerSide) continue;
						colorGrid[(size_t)gy * (tilesPerSide + 1) + gx] = tp.TileColors[ts + tt * (tp.OrderS + 1)];
					}
		}
		printf("template tiles: %u of %u cell tiles taken from %u patch(es)\n", placed, (uint)tileGrid.size(), (uint)tplInfo.Patchs.size());
	}

	uint16 zoneId = tplZone.getZoneId();
	if (!zoneName.empty())
	{
		// NeL's zone id is the cell index: row-major over the 256-column world grid.
		sint row = 0; char col[8] = { 0 };
		if (sscanf(zoneName.c_str(), "%d_%7s", &row, col) == 2)
		{
			// NeL's column is base 26 with 'A' = 0 in every position: "JR" is 9*26 + 17 = 251,
			// which is the id 21499 that 84_JR carries. It is not the spreadsheet column rule.
			sint c = 0;
			for (const char *p = col; *p; ++p) c = c * 26 + (toupper(*p) - 'A');
			zoneId = (uint16)((row - 1) * 256 + c);
		}
	}

	printf("template %s: patch 0 is order %ux%u, %u tiles, %u lumels/tile\n",
		CFile::getFilename(tpl).c_str(), (uint)src.OrderS, (uint)src.OrderT, srcTiles, lumelPerTile);
	{
		const CBezierPatch &b0 = src.Patch;
		CVector n = (b0.Vertices[1] - b0.Vertices[0]) ^ (b0.Vertices[3] - b0.Vertices[0]);
		printf("template patch0 V0(%.1f,%.1f) V1(%.1f,%.1f) V3(%.1f,%.1f) normalz %.2f\n",
			b0.Vertices[0].x, b0.Vertices[0].y, b0.Vertices[1].x, b0.Vertices[1].y,
			b0.Vertices[3].x, b0.Vertices[3].y, n.z);
	}
	printf("building %u x %u patches of %.1f m (order %u) over %.1f m, zone id %u\n",
		P, P, patchSize, order, cellSize, zoneId);

	std::vector<CPatchInfo> patchs(P * P);
	for (uint py = 0; py < P; ++py)
		for (uint px = 0; px < P; ++px)
		{
			CPatchInfo &pi = patchs[py * P + px];
			pi = src;                       // tile material, flags, noise rotation
			pi.OrderS = (uint8)order;
			pi.OrderT = (uint8)order;
			pi.ErrorSize = src.ErrorSize * (patchSize * patchSize) / ((float)src.OrderS * 2.f * (float)src.OrderT * 2.f);

			// tiles and colours for this patch's slice of the cell's tile grid
			pi.Tiles.resize((size_t)order * order);
			for (uint tt = 0; tt < order; ++tt)
				for (uint ts = 0; ts < order; ++ts)
					pi.Tiles[ts + tt * order] = tileGrid[(size_t)(py * order + ts) * tilesPerSide + (px * order + tt)];
			pi.TileColors.resize(((size_t)order + 1) * (order + 1));
			for (uint tt = 0; tt <= order; ++tt)
				for (uint ts = 0; ts <= order; ++ts)
					pi.TileColors[ts + tt * (order + 1)] = colorGrid[(size_t)(py * order + ts) * (tilesPerSide + 1) + (px * order + tt)];
			if (lumelPerTile) pi.Lumels.assign((size_t)order * order * lumelPerTile, 255);
			else pi.Lumels.clear();
			if (liPerCorner) pi.TileLightInfluences.assign(((size_t)order + 1) * (order + 1) * liPerCorner, src.TileLightInfluences[0]);
			else pi.TileLightInfluences.clear();

			float lx = (float)px * patchSize, ly = (float)py * patchSize;  // local metres
			float w3 = patchSize / 3.f;

			// Fit the 4x4 control grid to the height field. Corner values are interpolated
			// exactly; edge points carry the first derivative and interiors the cross term,
			// which is the Bezier form of a C1 bicubic Hermite surface.
			CCtrl ctrl;
			float f[2][2], fx[2][2], fy[2][2], fxy[2][2];
			for (uint cy = 0; cy < 2; ++cy)
				for (uint cx = 0; cx < 2; ++cx)
					hf.sample(lx + (float)cx * patchSize, ly + (float)cy * patchSize,
						f[cx][cy], fx[cx][cy], fy[cx][cy], fxy[cx][cy]);

			// The template's patches run s along +Y and t along +X, with Vertices[0] at the
			// cell-minimum corner. Writing them the other way round makes every surface normal
			// point down, and build_rbank then tessellates the zone to an empty retriever.
			for (uint i = 0; i < 4; ++i)        // i is s, along +Y
				for (uint j = 0; j < 4; ++j)    // j is t, along +X
				{
					uint cy = (i >= 2) ? 1 : 0, cx = (j >= 2) ? 1 : 0;   // nearest corner
					float sy = (i == 0 || i == 3) ? 0.f : ((i == 1) ? 1.f : -1.f);
					float sx = (j == 0 || j == 3) ? 0.f : ((j == 1) ? 1.f : -1.f);
					float z = f[cx][cy] + sx * w3 * fx[cx][cy] + sy * w3 * fy[cx][cy] + sx * sy * w3 * w3 * fxy[cx][cy];
					ctrl.C[i][j].set(ox + lx + (float)j * w3, oy + ly + (float)i * w3, z);
				}

			CBezierPatch &b = pi.Patch;
			b.Vertices[0] = ctrl.C[0][0]; b.Vertices[1] = ctrl.C[0][3];
			b.Vertices[2] = ctrl.C[3][3]; b.Vertices[3] = ctrl.C[3][0];
			b.Tangents[0] = ctrl.C[0][1]; b.Tangents[1] = ctrl.C[0][2];
			b.Tangents[2] = ctrl.C[1][3]; b.Tangents[3] = ctrl.C[2][3];
			b.Tangents[4] = ctrl.C[3][2]; b.Tangents[5] = ctrl.C[3][1];
			b.Tangents[6] = ctrl.C[2][0]; b.Tangents[7] = ctrl.C[1][0];
			b.Interiors[0] = ctrl.C[1][1]; b.Interiors[1] = ctrl.C[1][2];
			b.Interiors[2] = ctrl.C[2][2]; b.Interiors[3] = ctrl.C[2][1];

			// shared corner vertices, (P+1)^2 of them
			pi.BaseVertices[0] = (uint16)(py * (P + 1) + px);
			pi.BaseVertices[1] = (uint16)(py * (P + 1) + px + 1);
			pi.BaseVertices[2] = (uint16)((py + 1) * (P + 1) + px + 1);
			pi.BaseVertices[3] = (uint16)((py + 1) * (P + 1) + px);

			for (uint e = 0; e < 4; ++e) pi.BindEdges[e] = CPatchInfo::CBindInfo();
			// edge n links vertices n and (n+1)%4: 0 south, 1 east, 2 north, 3 west
			struct { sint dx, dy; uint theirEdge; } nb[4] = { { 0, -1, 2 }, { 1, 0, 3 }, { 0, 1, 0 }, { -1, 0, 1 } };
			for (uint e = 0; e < 4; ++e)
			{
				sint nx = (sint)px + nb[e].dx, ny = (sint)py + nb[e].dy;
				if (nx < 0 || ny < 0 || nx >= (sint)P || ny >= (sint)P) continue;   // zone border: no neighbour
				pi.BindEdges[e].NPatchs = 1;
				pi.BindEdges[e].ZoneId = zoneId;
				pi.BindEdges[e].Next[0] = (uint16)(ny * P + nx);
				pi.BindEdges[e].Edge[0] = (uint8)nb[e].theirEdge;
			}
			pi.Flags = 0;   // smooth across every edge
			for (uint c = 0; c < 4; ++c) pi.setCornerSmoothFlag(c, true);
		}

	CZone zone;
	std::vector<CBorderVertex> border;      // isolated cell: welding adds these later
	zone.build(zoneId, patchs, border, (P + 1) * (P + 1));

	{
		COFile f;
		if (!f.open(out)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
		f.serial(zone);
	}
	printf("wrote %s: %u patches, %u vertices\n", out.c_str(), (uint)patchs.size(), (P + 1) * (P + 1));

	// Measure the fit against the source grid, through the patches we just wrote.
	double maxErr = 0.0, sumErr = 0.0; uint n = 0;
	FILE *rep = report.empty() ? NULL : fopen(report.c_str(), "w");
	if (rep) fprintf(rep, "x,y,source_z,patch_z,error\n");
	for (uint iy = 0; iy < grid; ++iy)
		for (uint ix = 0; ix < grid; ++ix)
		{
			float lx = (float)ix * hf.Step, ly = (float)iy * hf.Step;
			float gt = lx / patchSize, gs = ly / patchSize;   // t is x, s is y
			uint px = std::min<uint>(P - 1, (uint)gt), py = std::min<uint>(P - 1, (uint)gs);
			float t = gt - (float)px, s = gs - (float)py;
			CVector v = patchs[py * P + px].Patch.eval(s, t);
			double e = v.z - hf.at((sint)ix, (sint)iy);
			if (fabs(e) > maxErr) maxErr = fabs(e);
			sumErr += fabs(e); ++n;
			if (rep) fprintf(rep, "%.3f,%.3f,%.4f,%.4f,%.4f\n", ox + lx, oy + ly, hf.at((sint)ix, (sint)iy), v.z, e);
		}
	if (rep) fclose(rep);
	printf("fit against the source grid: %u points, max |error| %.4f m, mean %.4f m\n", n, maxErr, sumErr / n);
	return 0;
}
