#ifndef _FEATURELINE_HPP_
#define _FEATURELINE_HPP_

#define FL_POINT_HIT_RADIUS 20.0f

#include <vector>
#include <Shader.hpp>
#include <IRenderable.hpp>
#include <IShaderRenderable.hpp>
#include <IRenderer.hpp>
#include <FeatureLineShader.hpp>
#include <SimplePointShader.hpp>
using std::vector;

class FeatureLine;

class FL_Point : public IRenderable
{
public:
    float x;
    float y;

    FL_Point(float x, float y);
    virtual ~FL_Point();

    void update();

    void updateParent();

    float squareDistanceTo(float fromX, float fromY);

    bool is_hit = false;

    FeatureLine *parentLine;

    virtual void render() final;

};

class FeatureLine : public IRenderable
{
public:
    FeatureLine() = delete;
    FeatureLine(float x1, float y1, float x2, float y2);
    FeatureLine(FL_Point *start, FL_Point *end);
    virtual ~FeatureLine();

    virtual void render() final;

    FL_Point *start;
    FL_Point *end;

    FL_Point *hit(float x, float y);

    void update();
    void updateAll();



private:

    void _generateBuffers();

};

struct FL_Pair
{
    FeatureLine *first;
    FeatureLine *second;
};


class FeatureLineManager : public IRenderer
{
public:
    static FeatureLineManager *getInstance();

    /**
     * Hit an endpoint of a featureline.
     */
    FL_Point *hit(int x, int y);

    /**
     * @param first Whether the first or second set of FeatureLine are active
     */
    void setActiveSet(bool first);

    void add(float x1, float y1, float x2, float y2);

    void add(FL_Point *start, FL_Point *end);

    void remove(FL_Point *point);

    void addLinePair(FeatureLine *first, FeatureLine *second);

    void addPoint(float x, float y);

    /**
     * Deletes all FeatureLine instances and resets the state.
     */
    void reset();

    /**
     * Handles the dragging of FL_Point
     */
    void handleDragging();

    virtual void render() final;



private:
    FeatureLineManager();
    ~FeatureLineManager();


    vector<FL_Pair *> line_pairs;
    static FeatureLineManager *instance;

    FL_Point *current_orphan;

    FL_Point *currently_dragging;

    bool activeSet = false;

    friend class MorphFile;
    friend class MorphFileReader;
    friend class MorphFileWriter;


};

#endif
