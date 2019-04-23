#import <iostream>
#import <vector>

#import <OpenGL/gl.h>

#import "NBodyPreferences.h"
#import "NBodyConstants.h"
#import "NBodyEngine.h"

#import "GLcontainers.h"
#import "GLUQuery.h"

#import "OpenGLView.h"

#pragma mark -

static const NSOpenGLPixelFormatAttribute kOpenGLAttribsLegacyProfile[7] =
{
    NSOpenGLPFADoubleBuffer,
    NSOpenGLPFAAccelerated,
    NSOpenGLPFAAcceleratedCompute,
    NSOpenGLPFAAllowOfflineRenderers,   // NOTE: Needed to connect to compute-only gpus
    NSOpenGLPFADepthSize, 24,
    0
};

static const NSOpenGLPixelFormatAttribute kOpenGLAttribsLegacyDefault[4] =
{
    NSOpenGLPFADoubleBuffer,
    NSOpenGLPFADepthSize, 24,
    0
};

@implementation OpenGLView {
    BOOL _fullscreen;
    
    NSDictionary* _options;
    NSOpenGLContext* _context;
    NSTimer* _timer;
    
    NBodyEngine* _engine;
    NBodyPreferences* _prefs;

    IBOutlet NSPanel* _panelHUD;
}

#pragma mark -
#pragma mark Private - Destructor

- (void) cleanUpOptions {
    if(_options){
        [_options release];
        _options = nil;
    }
}

- (void) cleanUpTimer{
    if(_timer){
        [_timer invalidate];
        [_timer release];
    }
}

- (void) cleanUpPrefs {
    if(_prefs){
        [_prefs addEntries:_engine.preferences];

        [_prefs write];
        [_prefs release];
        
        _prefs = nil;
    }
}

- (void) cleanUpEngine{
    if(_engine){
        [_engine release];
        _engine = nil;
    }
}

- (void) cleanUpObserver {
    // If self isn't removed as an observer, the Notification Center
    // will continue sending notification objects to the deallocated
    // object.
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

// Tear-down objects
- (void) cleanup {
    [self cleanUpOptions];
    [self cleanUpPrefs];
    [self cleanUpTimer];
    [self cleanUpEngine];
    [self cleanUpObserver];
}

#pragma mark -
#pragma mark Private - Utilities - Misc.

// When application is terminating cleanup the objects
- (void) quit:(NSNotification *)notification {
    [self  cleanup];
}

- (void) idle {
    [self setNeedsDisplay:YES];
}

- (void) toggleFullscreen {
    if(_prefs.fullscreen){
        [self enterFullScreenMode:[NSScreen mainScreen]
                      withOptions:_options];
    }
}

- (void) alert:(NSString *)pMessage{
    if(pMessage){
        NSAlert* pAlert = [NSAlert new];
        
        if(pAlert){
            [pAlert addButtonWithTitle:@"OK"];
            [pAlert setMessageText:pMessage];
            [pAlert setAlertStyle:NSCriticalAlertStyle];
            
            NSModalResponse response = [pAlert runModal];
            
            if(response == NSAlertFirstButtonReturn){
                NSLog(@">> MESSAGE: %@", pMessage);
            }
            
            [pAlert release];
        }
    }
}

- (BOOL) query {
    GLU::Query query;
    
    // NOTE: For OpenCL 1.2 support refer to <http://support.apple.com/kb/HT5942>
    GLstrings keys = {
         "120",   "130",  "285",  "320M",
        "330M", "X1800", "2400",  "2600",
        "3000",  "4670", "4800",  "4870",
        "5600",  "8600", "8800", "9600M"
    };
    
    std::cout << ">> N-body Simulation: Renderer = \"" << query.renderer() << "\"" << std::endl;
    std::cout << ">> N-body Simulation: Vendor   = \"" << query.vendor()   << "\"" << std::endl;
    std::cout << ">> N-body Simulation: Version  = \"" << query.version()  << "\"" << std::endl;
    
    return BOOL(query.match(keys));
}

- (NSOpenGLPixelFormat*) newPixelFormat {
    NSOpenGLPixelFormat* pFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:kOpenGLAttribsLegacyProfile];
    if(!pFormat){
        NSLog(@">> WARNING: Failed to initialize an OpenGL context with the desired pixel format!");
        NSLog(@">> MESSAGE: Attempting to initialize with a fallback pixel format!");
        
        pFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:kOpenGLAttribsLegacyDefault];
    }
    return pFormat;
}

#pragma mark - Private - Utilities - Prepare

- (void) preparePrefs {
    _prefs = [NBodyPreferences new];
    if(_prefs){
        _fullscreen = _prefs.fullscreen;
    }
}

- (void) prepareNBody {
    if([self query]){
        [self alert:@"Requires OpenCL 1.2!"];
        
        [self cleanUpOptions];
        [self cleanUpTimer];
        
        exit(-1);
    }else{
        NSRect frame = [[NSScreen mainScreen] frame];
        
        _engine = [[NBodyEngine alloc] initWithPreferences:_prefs];
        
        if(_engine){
            _engine.frame = frame;
            [_engine acquire];
        }
    }
}

- (void) prepareRunLoop {
    _timer = [[NSTimer timerWithTimeInterval:0.0
                                       target:self
                                     selector:@selector(idle)
                                     userInfo:self
                                      repeats:true] retain];
    
    [[NSRunLoop currentRunLoop] addTimer:_timer
                                 forMode:NSRunLoopCommonModes];
}

#pragma mark - Public - Designated Initializer

- (instancetype) initWithFrame:(NSRect)frameRect {
    BOOL bIsValid = NO;
    
    NSOpenGLPixelFormat* pFormat = [self newPixelFormat];
    
    if(pFormat) {
        self = [super initWithFrame:frameRect
                        pixelFormat:pFormat];
        
        if(self) {
            _context = [self openGLContext];
            bIsValid  = (_context != nil);
            
            _options = [[NSDictionary dictionaryWithObject:@(YES)
                                                    forKey:NSFullScreenModeSetting] retain];
            
            // It's important to clean up our rendering objects before we terminate -- Cocoa will
            // not specifically release everything on application termination, so we explicitly
            // call our cleanup (private object destructor) routines.
            [[NSNotificationCenter defaultCenter] addObserver:self
                                                     selector:@selector(quit:)
                                                         name:@"NSApplicationWillTerminateNotification"
                                                       object:NSApp];
        }else{
            NSLog(@">> ERROR: Failed to initialize an OpenGL context with attributes!");
        }
        
        [pFormat release];
    } else{
        NSLog(@">> ERROR: Failed to acquire a valid pixel format!");
    }
    
    if(!bIsValid){
        exit(-1);
    }
    
    return self;
}

#pragma mark - Public - Destructor

- (void) dealloc {
    [self cleanup];
    
    [super dealloc];
}

#pragma mark - Public - Prepare

- (void) prepareOpenGL {
    [super prepareOpenGL];
    
    [self preparePrefs];
    [self prepareNBody];
    [self prepareRunLoop];
    
    [self toggleFullscreen];
}

#pragma mark - Public - Delegates

- (BOOL) isOpaque{
    return YES;
}

- (BOOL) acceptsFirstResponder {
    return YES;
}

- (BOOL) becomeFirstResponder{
    return  YES;
}

- (BOOL) resignFirstResponder{
    return YES;
}

- (BOOL) applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)theApplication{
    return YES;
}

#pragma mark - Public - Updates

//- (void) renewGState{
//    [super renewGState];
//    [[self window] disableScreenUpdatesUntilFlush];
//}

#pragma mark - Public - Display

- (void) resize {
    if(_engine){
        NSRect bounds = [self bounds];
        
        [_engine resize:bounds];
    }
}

- (void) reshape {
    [super reshape];
    [self resize];
}

- (void) drawRect:(NSRect)dirtyRect {
    [_engine draw];
}

#pragma mark - Public - Help

- (IBAction) toggleHelp:(id)sender {
    if([_panelHUD isVisible]){
        [_panelHUD orderOut:sender];
    } else {
        [_panelHUD makeKeyAndOrderFront:sender];
    }
}

#pragma mark - Public - Fullscreen

- (IBAction) toggleFullscreen:(id)sender {
    if([self isInFullScreenMode]){
        [self exitFullScreenModeWithOptions:_options];
        [[self window] makeFirstResponder:self];
        _prefs.fullscreen = NO;
    }else{
        [self enterFullScreenMode:[NSScreen mainScreen]
                      withOptions:_options];
        _prefs.fullscreen = YES;
    }
}

#pragma mark -
#pragma mark Public - Keys

- (void) keyDown:(NSEvent *)event {
    if(event){
        NSString* pChars = [event characters];
        
        if([pChars length]) {
            unichar key = [[event characters] characterAtIndex:0];
            
            if(key == 27) {
                [self toggleFullscreen:self];
            } else{
                _engine.command = key;
            }
        }
    }
}

- (void) mouseDown:(NSEvent *)event {
    if(event){
        NSPoint where  = [event locationInWindow];
        NSRect  bounds = [self bounds];
        NSPoint point  = NSMakePoint(where.x, bounds.size.height - where.y);
        
        [_engine click:NBody::Mouse::Button::kDown
                  point:point];
    }
}

- (void) mouseUp:(NSEvent *)event {
    if(event){
        NSPoint where  = [event locationInWindow];
        NSRect  bounds = [self bounds];
        NSPoint point  = NSMakePoint(where.x, bounds.size.height - where.y);
        
        [_engine click:NBody::Mouse::Button::kUp
                  point:point];
    }
}

- (void) mouseDragged:(NSEvent *)event {
    if(event){
        NSPoint where = [event locationInWindow];
        
        where.y = 1080.0f - where.y;
        
        [_engine move:where];
    }
}

- (void) scrollWheel:(NSEvent *)event{
    if(event){
        CGFloat dy = [event deltaY];
        [_engine scroll:dy];
    }
}

@end
